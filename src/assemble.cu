#include <type_traits>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/async/transform.h>

#include <chrono>

#include "alloc.h"
#include "indexing.h"
#include "Mesh.h"
#include "Field.h"
#include "Array.h"
#include "csr.h"
#include "matrix.h"

#include "assemble.h"

#define kRHOC (0.5)
#define kDT (1e-1)
#define kALPHAM ((3.0 - kRHOC) / (1.0 + kRHOC))
#define kALPHAF (1.0 / (1.0 + kRHOC))
#define kGAMMA (0.5 + kALPHAM - kALPHAF)

#define NQR (4)
#define BS (6)
#define M2D(aa, ii) ((aa) * BS + (ii))
#define M4D(aa, bb, ii, jj) \
	(((aa) * (NSHL) + ii) * (NSHL) * BS + (bb) * BS + (jj))

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define kRHO (1.0e3)
#define kCP (4.2e3)
#define kKAPPA (0.66)
#define kMU (1.0e-3)

__constant__ f64 fb[3] = {0.0, 0.0, -9.81};
__constant__ f64 gw[4] = {0.0416666666666667, 0.0416666666666667, 0.0416666666666667, 0.0416666666666667};
__constant__ f64 shlu[16] = {0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105,
														 0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105,
														 0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105,
														 0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685};

__constant__ f64 shlgradu[12] = {-1.0, -1.0, -1.0,
																 1.0, 0.0, 0.0,
																 0.0, 1.0, 0.0,
																 0.0, 0.0, 1.0};

const f64 h_gw[4] = {0.0416666666666667, 0.0416666666666667, 0.0416666666666667, 0.0416666666666667};
const f64 h_shlu[16] = {0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105,
											0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105,
											0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105,
											0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685};

const f64 h_shlgradu[12] = {-1.0, -1.0, -1.0,
													 1.0, 0.0, 0.0,
													 0.0, 1.0, 0.0,
													 0.0, 0.0, 1.0};

/** Load the value from the source array to the destination array belonging to the batch
 * The source array is columned-majored of size (max(elem_size, stridex), num_node)
 * The destination array is columned-majored of size (max(NSHL * BS, stridey), batch_size)
 * @tparam I The type of the index
 * @tparam T The type of the value
 * @tparam NSHL The number of shape functions
 * @param[in] batch_size The number of elements in the batch
 * @param[in] elem_size The size of the element
 * @param[in] batch_index_ptr[:] The index of the elements in the batch
 * @param[in] ien[NSHL, :] The index of the nodes in the element
 * @param[in] x[ldx, :] The source array to be loaded
 * @param[in] stridex The increment of the source array, stridex >= elem_size
 * @param[out] y[ldy, :] The destination array to be loaded
 * @param[in] stridey The increment of the destination array, stridey >= BS * NSHL
 */
template<typename I, typename T, I NSHL=4>
__global__ void LoadElementValueKernel(I batch_size, int elem_size,
																			 const I* batch_index_ptr,
																			 const I* ien,
																			 const T* x, int stridex,
																			 T* y, int stridey) {

	I idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= batch_size * NSHL) return;
	I i = idx / NSHL, lane = idx % NSHL;
	I iel = batch_index_ptr[i];
	I node_id = ien[iel * NSHL + lane];

	y += i * stridey;
	x += node_id * stridex;

	for (I j = 0; j < elem_size; ++j) {
		y[j * NSHL + lane] = x[j];
	}
}


template<typename I, typename T, I bs=1, I NSHL=4>
__global__ void LoadElementValueAXPYKernel(I n, const I* ien, const I* index, const T* src, T* dst, T alpha) {
	I idx = blockIdx.x * blockDim.x + threadIdx.x;
	I num_thread = blockDim.x * gridDim.x;
	while(idx < n) {
		#pragma unroll
		for(I i = 0; i < NSHL; ++i) {
			I node_id = ien[index[idx] * NSHL + i];
			#pragma unroll
			for(I j = 0; j < bs; ++j) {
				dst[idx * bs + j] = fma(src[node_id * bs + j], alpha, dst[idx * bs + j]);
			}
		}
		idx += num_thread;
	}
}

/** Add the elementwise residual vector to the global residual vector
 * This is the CUDA kernel for the function ElemRHSLocal2Global
 * @tparam I The type of the index
 * @tparam T The type of the value
 * @tparam NSHL The number of shape functions
 * @param[in]     batch_size The number of elements in the batch
 * @param[in]     batch_index_ptr[:] The index of the elements in the batch
 * @param[in]     ien[NSHL, :] The index of the nodes in the element
 * @param[in]     width The width of the elementwise residual vector
 * @param[in]     elem_F[lda, NSHL * batch_size] The elementwise residual vector
 * @param[in]     lda The leading dimension of the elem_F
 * @param[in,out] F[ldb, :] The global residual vector
 * @param[in]     ldb The block size
 */
template <typename I, typename T, I NSHL=4>
__global__ void
ElemRHSLocal2GlobalKernel(I batch_size, const I* batch_index_ptr, const I* ien,
													I width,
													const T* elem_F, I lda,
													T* F, I ldb) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batch_size * NSHL) {
		return;
	}
	int iel = batch_index_ptr[i / NSHL];
	int node_id = ien[iel * NSHL + i % NSHL];
	elem_F += i * lda;

	for(I j = 0; j < width; ++j) {
		F[node_id * ldb + j] += elem_F[j];
	}
}

/** Add the elementwise residual vector to the global residual vector
 * @tparam I The type of the index
 * @tparam T The type of the value
 * @tparam NSHL The number of shape functions
 * @param[in]     batch_size The number of elements in the batch
 * @param[in]     batch_index_ptr[:] The index of the elements in the batch
 * @param[in]     ien[NSHL, :] The index of the nodes in the element
 * @param[in]     elem_F[BS, NSHL * batch_size] The elementwise residual vector
 * @param[in]     lda The leading dimension of the elem_F
 * @param[in,out] F[:] The global residual vector
 * @param[in]     ldb The block size
 */
template <typename I, typename T, I NSHL=4>
void ElemRHSLocal2Global(I batch_size, const I* batch_index_ptr, const I* ien,
												 I width,
												 const T* elem_F, I lda,
												 T* F, I ldb) {
	int block_dim = 256;
	int grid_dim = CEIL_DIV(batch_size * NSHL, block_dim);
	ElemRHSLocal2GlobalKernel<I, T><<<grid_dim, block_dim>>>(batch_size, batch_index_ptr, ien,
																													 width,
																													 elem_F, lda,
																													 F, ldb);
}

template <typename I, I NSHL=4>
__global__ void
GetElemBatched(I batch_size, const I* batch_index_ptr, const I* ien, I* batch_ind) {
	extern __shared__ I shared[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	if(idx >= batch_size * NSHL) return;



	int iel = batch_index_ptr[idx / NSHL];

	shared[tid] = ien[iel * NSHL + idx % NSHL];

	__syncthreads();


}


template <typename I, typename T, I NSHL=4>
void ElemLHSLocal2GlobalBlocked(I batch_size, const I* batch_index_ptr, const I* ien,
															  I block_row, I block_col,
															  const T* elem_J, int lda, int* stride,
															  Matrix* J, int ldb) {
	int block_dim = 256;
	int grid_dim = CEIL_DIV(batch_size * NSHL * NSHL, block_dim);

	I* batch_ind = (I*)CdamMallocDevice(sizeof(I) * batch_size * NSHL);
	GetElemBatched<I, NSHL><<<CEIL_DIV(batch_size * NSHL, block_dim), block_dim>>>(batch_size, batch_index_ptr, ien, batch_ind);


	CdamFreeDevice(batch_ind);

}


template<typename I, typename T, I NSHL=4>
__global__ void GetElemJ3DKernel(I batch_size, const I* ien, const I* batch_index_ptr, const T* xg, T* elemJ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= batch_size) return;

	I iel = batch_index_ptr[idx];

	const T* xg_ptr0 = xg + ien[iel * NSHL + 0] * 3;
	const T* xg_ptr1 = xg + ien[iel * NSHL + 1] * 3;
	const T* xg_ptr2 = xg + ien[iel * NSHL + 2] * 3;
	const T* xg_ptr3 = xg + ien[iel * NSHL + 3] * 3;
	
	T* elemJ_ptr = elemJ + idx * 10;

	elemJ_ptr[0] = xg_ptr1[0] - xg_ptr0[0];
	elemJ_ptr[1] = xg_ptr1[1] - xg_ptr0[1];
	elemJ_ptr[2] = xg_ptr1[2] - xg_ptr0[2];

	elemJ_ptr[3] = xg_ptr2[0] - xg_ptr0[0];
	elemJ_ptr[4] = xg_ptr2[1] - xg_ptr0[1];
	elemJ_ptr[5] = xg_ptr2[2] - xg_ptr0[2];

	elemJ_ptr[6] = xg_ptr3[0] - xg_ptr0[0];
	elemJ_ptr[7] = xg_ptr3[1] - xg_ptr0[1];
	elemJ_ptr[8] = xg_ptr3[2] - xg_ptr0[2];

}

template<typename I, typename T, I NSHL=4>
__global__ void GetElemDetJKernel(I ne, T* elem_metric) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= ne) return;

	T* elem_lu = elem_metric + idx * 10; /* elem_lu is the LU decomposition of the Jacobian matrix */
	elem_lu[9] = fabs(elem_lu[0] * elem_lu[4] * elem_lu[8]);
}

template<typename I, typename T, I NSHL=4>
__global__ void CopyElemJacobiansKernel(I ne, const T** elemJ, T* elemJ_ptr) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= ne) return;

	const T* elemJ_ptr0 = elemJ[idx];
	T* elemJ_ptr1 = elemJ_ptr + idx * 10;
	elemJ_ptr1[0] = elemJ_ptr0[0];
	elemJ_ptr1[1] = elemJ_ptr0[1];
	elemJ_ptr1[2] = elemJ_ptr0[2];
	elemJ_ptr1[3] = elemJ_ptr0[3];
	elemJ_ptr1[4] = elemJ_ptr0[4];
	elemJ_ptr1[5] = elemJ_ptr0[5];
	elemJ_ptr1[6] = elemJ_ptr0[6];
	elemJ_ptr1[7] = elemJ_ptr0[7];
	elemJ_ptr1[8] = elemJ_ptr0[8];
}

template<typename I, typename T, I NSHL=4>
__global__ void
AssemleWeakFormKernelHeat(I batch_size, const T* elem_invJ,
													const T* qr_wgalpha, const T* qr_dwgalpha, const T* qr_wggradalpha,
													const T* shgradl,
													T* elem_F, T* elem_J) {


	I idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size * NSHL) return;
	I iel = idx / NSHL;
	I lane = idx % NSHL;


	const T* elem_invJ_ptr = elem_invJ + iel * 10;
	const T detJ = elem_invJ_ptr[9];
	const T* qr_wgalpha_ptr = qr_wgalpha + iel * NQR;
	const T* qr_dwgalpha_ptr = qr_dwgalpha + iel * NQR;
	const T* qr_wggradalpha_ptr = qr_wggradalpha + iel * 3;
	const T* shgradl_ptr = shgradl + iel * 3 * NSHL;

	/* Weak form: */
	/* F_A[0:NSHL] += (gw[0:NQR] * detJ * qr_dwgalpha_ptr[0:NQR]) @ shl[0:NQR, 0:NSHL] */
	/* F_A[0:NSHL] += sum(gw[0:NQR]) * detJ * qr_wggradalpha_ptr[0:3] @ shgradl[0:3, 0:NSHL] */

	if (elem_F) {
		T Fidx = 0.0;
		#pragma unroll
		for(I q = 0; q < NQR; ++q) {
			Fidx += gw[q] * detJ * qr_dwgalpha_ptr[q] * shlu[lane*NQR+q];
		}
		Fidx += 1.0/6.0 * detJ * (qr_wggradalpha_ptr[0] * shgradl[idx * 3 + 0] +
															qr_wggradalpha_ptr[1] * shgradl[idx * 3 + 1] +
															qr_wggradalpha_ptr[2] * shgradl[idx * 3 + 2]);
		elem_F[idx] = Fidx;
	}

	/* Weak form: */
	/* J_AB[0:NSHL, 0:NSHL] += kALPHAM * detJ * shl[0,NQR, NSHL].T @ (gw[0:NQR, None] * shl[0:NQR, 0:NSHL]) */
	/* J_AB[0:NSHL, 0:NSHL] += kDT * kALPHAF * kGAMMA * sum(gw[0:NQR]) * detJ 
													 * shgradl[0:3, 0:NSHL].T @ shgradl[0:3, 0:NSHL] */
	if (elem_J) {
		T* elem_J_ptr = elem_J + iel * NSHL * NSHL;
		T fact1 = kALPHAM;
		T fact2 = kDT * kALPHAF * kGAMMA;

		#pragma unroll
		for(I aa = 0; aa < NSHL; ++aa) {
			elem_J_ptr[aa * NSHL + lane] = 0.0;
		}

		#pragma unroll
		for(I aa = 0; aa < NSHL; ++aa) {
			#pragma unroll
			for(I q = 0; q < NQR; ++q) {
				elem_J_ptr[aa * NSHL + lane] += fact1 * detJ * gw[q] * shlu[NQR * aa + q] * shlu[lane * NQR + q];
			}
		}

		#pragma unroll
		for(I aa = 0; aa < NSHL; ++aa) {
			elem_J_ptr[aa * NSHL + lane] += fact2 * detJ * (1.0/6.0) * (shgradl_ptr[aa * 3 + 0] * shgradl_ptr[lane * 3 + 0] +
																																	shgradl_ptr[aa * 3 + 1] * shgradl_ptr[lane * 3 + 1] +
																																	shgradl_ptr[aa * 3 + 2] * shgradl_ptr[lane * 3 + 2]);
		}
	}
}
template <typename I, typename T>
__device__ void GetStabTau(const T* elem_invJ, const T* uadv, T rho, T cp, T mu, T kappa, T dt, T* tau) {
	T Ginv[9] = {0.0, 0.0, 0.0,
							 0.0, 0.0, 0.0,
							 0.0, 0.0, 0.0};
	T tau_tmp[3] = {0.0, 0.0, 0.0};

	/* Ginv = elem_invJ[0:3, 0:3] @ elem_invJ[0:3, 0:3].T */
	#pragma unroll
	for(I k = 0; k < 3; ++k) {
		#pragma unroll
		for(I i = 0; i < 3; ++i) {
			#pragma unroll
			for(I j = 0; j < 3; ++j) {
				Ginv[i * 3 + j] += elem_invJ[i + k * 3] * elem_invJ[j + k * 3];
			}
		}
	}

	/* tau_tmp[0] = 4.0 / dt^2 */
	tau_tmp[0] = 4.0 / (dt * dt);
	/* tau_tmp[1] = uadv[0:3] @ Ginv[0:3, 0:3] @ uadv[0:3] */
	/* tau_tmp[2] = Ginv[0:3, 0:3] @ Ginv[0:3, 0:3] */
	#pragma unroll
	for(I i = 0; i < 3; ++i) {
		#pragma unroll
		for(I j = 0; j < 3; ++j) {
			tau_tmp[1] += Ginv[i * 3 + j] * uadv[i] * uadv[j];
			tau_tmp[2] += Ginv[i * 3 + j] * Ginv[i * 3 + j];
		}
	}

	mu /= rho;
	kappa /= rho * cp;
	/* tauM */
	tau[0] = 1 / rho / sqrt(tau_tmp[0] + tau_tmp[1] + 3.0 * mu * mu * tau_tmp[2]);
	/* tauC */
	tau[1] = sqrt(tau_tmp[1] + 3.0 * mu * mu * tau_tmp[2]) / (Ginv[0] + Ginv[4] + Ginv[8]);
	/* tauPhi */
	tau[2] = 1.0 / sqrt(tau_tmp[0] + tau_tmp[1]);
	/* tauT */
	tau[3] = 1.0 / (rho * cp * sqrt(tau_tmp[0] + tau_tmp[1] + 3.0 * kappa * kappa * tau_tmp[2]));
}

template<typename I, typename T, I NSHL=4>
__device__ void GetrLi(const T* du, const T* uadv, const T* gradu, const T* gradp, T rho, T mu, T* rLi) {
	#pragma unroll
	for(I i = 0; i < 3; ++i) {
		rLi[i] = du[i] - fb[i] + uadv[0] * gradu[i*3+0] + uadv[1] * gradu[i*3+1] + uadv[2] * gradu[i*3+2];
		rLi[i] = rho * rLi[i] + gradp[i];
	}
}


template<typename I, typename T, I NSHL=4>
__global__ void
AssembleWeakFormKernel(I batch_size,
											 const T* elem_invJ, const T* shgradg,
											 const T* qr_wgalpha, const T* qr_dwgalpha, const T* qr_wggradalpha,
											 T* elem_F, T* elem_J) {
	I iel = blockIdx.x * blockDim.x + threadIdx.x;
	if(iel >= batch_size) return;

	elem_invJ += iel * 10;
	T detJ = elem_invJ[9];
	shgradg += iel * NSHL * 3;
	/* Column-majored */
	/* [:, 0:3]: Velocity */
	/* [:, 3]: Pressure */
	/* [:, 4]: Phi */
	/* [:, 5]: Temperature */
	qr_wgalpha += iel * NQR * BS; /* (NQR, BS) */
	qr_dwgalpha += iel * NQR * BS; /* (NQR, BS) */
	qr_wggradalpha += iel * 3 * BS; /* (3, BS) */

	if(elem_F) {
		elem_F += iel * NSHL * BS;
	}

	if(elem_J) {
		elem_J += iel * NSHL * NSHL * BS * BS;
		for(I i = 0; i < NSHL * NSHL * BS * BS; ++i) {
			elem_J[i] = 0.0;
		}
	}

	T tau[4] = {0.0, 0.0, 0.0, 0.0}; 
	T divu = qr_wggradalpha[0] + qr_wggradalpha[4] + qr_wggradalpha[8];
	
	T rLi[3];
	T uadv[3];
	T ufull[3];
	T shconv[NSHL];

	for(I iq = 0; iq < NQR; ++iq) {
		/* Get uadv */
		uadv[0] = qr_wgalpha[BS * 0 + iq];
		uadv[1] = qr_wgalpha[BS * 1 + iq];
		uadv[2] = qr_wgalpha[BS * 2 + iq];

		/* Get rLi */
		for(I i = 0; i < 3; ++i) {
			rLi[i] = qr_dwgalpha[BS * i + iq] - fb[i];
			rLi[i] += uadv[0] * qr_wggradalpha[BS * i + 0];
			rLi[i] += uadv[1] * qr_wggradalpha[BS * i + 1];
			rLi[i] += uadv[2] * qr_wggradalpha[BS * i + 2];
			rLi[i] = kRHO * rLi[i] + qr_wggradalpha[BS * 3 + iq];
		}

		/* Get Stab */
		GetStabTau<I, T>(elem_invJ, uadv, kRHO, kCP, kMU, kKAPPA, kDT, tau);

		for(I aa = 0; aa < NSHL; ++aa) {
			shconv[aa] = uadv[0] * shgradg[aa * 3 + 0] + uadv[1] * shgradg[aa * 3 + 1] + uadv[2] * shgradg[aa * 3 + 2];
		}

		if(elem_F) {
			T tmp0[3];
			T tmp1[3 * 3];
			T tmp2[1];
			/* Get tmp0 */
			for(I i = 0; i < 3; ++i) {
				tmp0[i] = 0.0;
				tmp0[i] += qr_dwgalpha[BS * i + iq] - fb[i];
				tmp0[i] += (ufull[0] - tau[0] * rLi[0]) * qr_wggradalpha[BS * i + 0];
				tmp0[i] += (ufull[1] - tau[0] * rLi[1]) * qr_wggradalpha[BS * i + 1];
				tmp0[i] += (ufull[2] - tau[0] * rLi[2]) * qr_wggradalpha[BS * i + 2];
				tmp0[i] *= kRHO;
			}

			/* Get tmp1 */
			for(I i = 0; i < 3; ++i) {
				for(I j = 0; j < 3; ++j) {
					tmp1[i * 3 + j] = 0.0;
					tmp1[i * 3 + j] += kMU * (qr_wggradalpha[BS * i + j] * qr_wggradalpha[BS * j + i]);
					tmp1[i * 3 + j] += kRHO * tau[0] * rLi[i] * qr_wgalpha[BS * j + iq];
					tmp1[i * 3 + j] -= kRHO * tau[0] * tau[0] * rLi[i] * rLi[j];
				}
			}

			/* Get tmp2 */
			tmp2[0] = -qr_wgalpha[BS * 3 + iq] + kRHO * tau[1] * divu;

			/* Momentum */
			#pragma unroll
			for(I aa = 0; aa < NSHL; ++aa) {
				#pragma unroll
				for(I ii = 0; ii < 3; ++ii) {
					T bm = 0.0;
					bm += shlu[aa * NQR + iq] * tmp0[ii];
					bm += shgradg[aa * 3 + 0] * tmp1[ii * 3 + 0];
					bm += shgradg[aa * 3 + 1] * tmp1[ii * 3 + 1];
					bm += shgradg[aa * 3 + 2] * tmp1[ii * 3 + 2];
					bm += shgradg[aa * 3 + ii] * tmp2[0];
					elem_F[M2D(aa, ii)] = bm * gw[iq] * detJ;
				}
			}
			/* Continuity */
			for(I aa = 0; aa < NSHL; ++aa) {
				T bc = 0.0;
				bc += shlu[aa * NQR + iq] * divu;
				bc += tau[0] * rLi[0] * shgradg[aa * 3 + 0];
				bc += tau[0] * rLi[1] * shgradg[aa * 3 + 1];
				bc += tau[0] * rLi[2] * shgradg[aa * 3 + 2];
				elem_F[M2D(aa, 3)] = bc * gw[iq] * detJ;
			}
			/* Phi */
			for(I aa = 0; aa < NSHL; ++aa) {
				T bp = qr_dwgalpha[BS * 4 + iq]
						 + uadv[0] * qr_wggradalpha[BS * 4 + 0]
						 + uadv[1] * qr_wggradalpha[BS * 4 + 1]
						 + uadv[2] * qr_wggradalpha[BS * 4 + 2];
				elem_F[M2D(aa, 4)] = bp * (shlu[aa * NQR + iq] + tau[2] * shconv[aa]) * gw[iq] * detJ;
			}

			/* Temperature */
			for(I aa = 0; aa < NSHL; ++aa) {
				T bt = kRHO * kCP * (qr_dwgalpha[BS * 5 + iq]
						 + uadv[0] * qr_wggradalpha[BS * 5 + 0]
						 + uadv[1] * qr_wggradalpha[BS * 5 + 1]
						 + uadv[2] * qr_wggradalpha[BS * 5 + 2]) 
						 * (shlu[aa * NQR + iq] + kRHO * kCP * tau[3] * shconv[aa]);

				bt += kKAPPA * (qr_wggradalpha[BS * 5 + 0] * shgradg[aa * 3 + 0] +
												qr_wggradalpha[BS * 5 + 1] * shgradg[aa * 3 + 1] +
												qr_wggradalpha[BS * 5 + 2] * shgradg[aa * 3 + 2]);

				elem_F[M2D(aa, 5)] = bt * gw[iq] * detJ;
			}
		}

		if(elem_J) {
			f64 fact1 = kALPHAM;
			f64 fact2 = kDT * kALPHAF * kGAMMA;

			T tmp[NSHL][NSHL];
			for(I aa = 0; aa < NSHL; aa++) {
				for(I bb = 0; bb < NSHL; bb++) {
					tmp[aa][bb] = 0.0;
					tmp[aa][bb] += fact1 * kRHO * shlu[aa * NQR + iq] * shlu[bb * NQR + iq];
					tmp[aa][bb] += fact1 * kRHO * tau[0] * shconv[aa] * shlu[bb * NQR + iq];
					tmp[aa][bb] += fact2 * shlu[aa * NQR + iq] * shconv[bb];
					tmp[aa][bb] += fact2 * tau[0] * shconv[aa] * shconv[bb];
					tmp[aa][bb] += fact2 * kMU * (shgradg[aa * 3 + 0] * shgradg[bb * 3 + 0] +
																				shgradg[aa * 3 + 1] * shgradg[bb * 3 + 1] +
																				shgradg[aa * 3 + 2] * shgradg[bb * 3 + 2]);
				}
			}

			for(I aa = 0; aa < NSHL; ++aa) {
				for(I bb = 0; bb < NSHL; ++bb) {
					/* dRM/dU */
					elem_J[M4D(aa, bb, 0, 0)] += tmp[aa][bb] * gw[iq] * detJ;
					elem_J[M4D(aa, bb, 1, 1)] += tmp[aa][bb] * gw[iq] * detJ;
					elem_J[M4D(aa, bb, 2, 2)] += tmp[aa][bb] * gw[iq] * detJ;
					for(I ii = 0; ii < 3; ++ii) {
						for(I jj = 0; jj < 3; ++jj) {
							elem_J[M4D(aa, bb, ii, jj)] += fact2 * kMU * shgradg[aa * 3 + jj] * shgradg[bb * 3 + ii] * gw[iq] * detJ;
							elem_J[M4D(aa, bb, ii, jj)] += fact2 * kRHO * tau[1] * shgradg[aa * 3 + ii] * shgradg[bb * 3 + jj] * gw[iq] * detJ;
						}
					}

					/* dRC/dU */
					for(I ii = 0; ii < 3; ++ii) {
						elem_J[M4D(aa, bb, 3, ii)] += fact1 * kRHO * shgradg[aa * 3 + ii] * shlu[bb * NQR + iq] * gw[iq] * detJ;
						elem_J[M4D(aa, bb, 3, ii)] += fact2 * shlu[aa * NQR + iq] * shgradg[bb * 3 + ii] * gw[iq] * detJ;
						elem_J[M4D(aa, bb, 3, ii)] += fact2 * tau[0] * shgradg[aa * 3 + ii] * shconv[bb] * gw[iq] * detJ;
					}

					/* dRM/dP */
					for(I ii = 0; ii < 3; ++ii) {
						elem_J[M4D(aa, bb, ii, 3)] -= fact2 * kRHO * shgradg[aa * 3 + ii] * shlu[bb * NQR + iq] * gw[iq] * detJ;
						elem_J[M4D(aa, bb, ii, 3)] -= fact2 * tau[0] * shconv[aa] * shgradg[bb * 3 + ii] * gw[iq] * detJ;
					}

					/* dRC/dP */
					elem_J[M4D(aa, bb, 3, 3)] += fact2 * tau[0] * (shgradg[aa * 3 + 0] * shgradg[bb * 3 + 0] +
																													shgradg[aa * 3 + 1] * shgradg[bb * 3 + 1] +
																													shgradg[aa * 3 + 2] * shgradg[bb * 3 + 2]) * gw[iq] * detJ;

					/* dRphi/dphi */
					elem_J[M4D(aa, bb, 4, 4)] += (shlu[aa * NQR + iq] + tau[2] * shconv[aa]) 
																			* (fact1 * shlu[bb * NQR + iq] + fact2 * shconv[bb]) * gw[iq] * detJ;

					/* dRT/dT */
					elem_J[M4D(aa, bb, 5, 5)] += (shlu[aa * NQR + iq] + kRHO * kCP * tau[3] * shconv[aa]) 
																			* kRHO * kCP * (fact1 * shlu[bb * NQR + iq] + fact2 * shconv[bb]) * gw[iq] * detJ;
					elem_J[M4D(aa, bb, 5, 5)] += kKAPPA * (shgradg[aa * 3 + 0] * shgradg[bb * 3 + 0] +
																								 shgradg[aa * 3 + 1] * shgradg[bb * 3 + 1] +
																								 shgradg[aa * 3 + 2] * shgradg[bb * 3 + 2]) * gw[iq] * detJ;
				}
			}
		}
	}
	

}

template <typename I, typename T>
__global__ void GetElemInvJ3DLoadMatrixBatchKernel(I batch_size, T** A, T* elem_metric, I inc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batch_size) return;
	A[i] = elem_metric + i * inc;
}

template <typename I, typename T, typename INC>
__global__ void IncrementByN(I n, T* a, INC inc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n) return;
	a[i] += inc;
}


template <typename I, typename T, I NSHL=4>
void GetElemInvJ3D(I batch_size, const I* ien, const I* batch_index_ptr, const T* xg, T* elem_metric, void* buff, cublasHandle_t handle) {
	f64 one = 1.0, zero = 0.0;
	int block_size = 256;
	int num_block = CEIL_DIV(batch_size, block_size);

	cudaStream_t stream[1] = {0};
	// cudaStreamCreate(stream + 0);


	/* 0. Load the elements in the batch into elem_metric[0:10, :]  */
	GetElemJ3DKernel<I, T, NSHL><<<num_block, block_size, 0, stream[0]>>>(batch_size, ien, batch_index_ptr, xg, elem_metric);

	// cublasSetStream(handle, stream[0]);

	byte* buff_ptr = (byte*)buff;

	T* d_elem_buff = (T*)buff_ptr;
	T** d_mat_batch = (T**)(buff_ptr + batch_size * 10 * sizeof(T));
	T** d_matinv_batch = (T**)(buff_ptr + batch_size * 10 * sizeof(T) + batch_size * sizeof(T*));

	int* info = (int*)((byte*)d_matinv_batch + batch_size * sizeof(T*));
	int* pivot = info + batch_size;

	/* Set mat_batch */
	GetElemInvJ3DLoadMatrixBatchKernel<I, T><<<num_block, block_size, 0, stream[0]>>>(batch_size, d_matinv_batch, d_elem_buff, 10);
	GetElemInvJ3DLoadMatrixBatchKernel<I, T><<<num_block, block_size, 0, stream[0]>>>(batch_size, d_mat_batch, elem_metric, 10);	

	/* LU decomposition */
	cublasDgetrfBatched(handle, 3, d_mat_batch, 3, pivot, info, batch_size);
	/* Calculate the determinant of the Jacobian matrix */
	GetElemDetJKernel<<<num_block, block_size, 0, stream[0]>>>(batch_size, elem_metric);
	/* Invert the Jacobian matrix */
	cublasDgetriBatched(handle, 3, d_mat_batch, 3, pivot, d_matinv_batch, 3, info, batch_size);

	/* Copy the inverse of the Jacobian matrix to the elem_metric */
	cublasDgeam(handle,
							CUBLAS_OP_N, CUBLAS_OP_N,
							9, batch_size,
							&one,
							d_elem_buff, 10,
							&zero,
							elem_metric, 10,
							elem_metric, 10);

}


template<typename I, typename T>
__global__ void
GetShapeGradKernel(I num_elem, const T* elem_invJ, T* shgrad) {
	i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num_elem) return;

	const T* elem_invJ_ptr = elem_invJ + idx * 10;
	T* shgrad_ptr = shgrad + idx * 12;
	/* shgrad_ptr[0:3, 1:NSHL] = elem_invJ[0:3, 0:3] */
	#pragma unroll
	for(i32 i = 0; i < 3; ++i) {
		for(i32 j = 0; j < 3; ++j) {
			shgrad_ptr[i * 3 + j + 3] = elem_invJ_ptr[i + j * 3];
		}
	}

	/* shgrad_ptr[0:3, 0] = -sum(elem_invJ[0:3, 0:3], axis=1)*/
	shgrad_ptr[0] = -shgrad_ptr[3] - shgrad_ptr[6] - shgrad_ptr[9];
	shgrad_ptr[1] = -shgrad_ptr[4] - shgrad_ptr[7] - shgrad_ptr[10];
	shgrad_ptr[2] = -shgrad_ptr[5] - shgrad_ptr[8] - shgrad_ptr[11];
}


template<typename I, typename T>
void IntElemAssembly(I batch_size,
										 const T* elem_invJ, const T* shgradg,
										 const T* qr_wgalpha, const T* qr_dwgalpha, const T* qr_wggradalpha,
										 T* elem_F, T* elem_J) {
	i32 block_dim = 256;
	i32 grid_dim = CEIL_DIV(batch_size, block_dim);

	AssembleWeakFormKernel<I, T><<<grid_dim, block_dim>>>(batch_size,
																												elem_invJ, shgradg,
																												qr_wgalpha, qr_dwgalpha, qr_wggradalpha,
																												elem_F, elem_J);

}

__BEGIN_DECLS__





void AssembleSystemTet(Mesh3D *mesh,
											 /* Field *wgold, Field* dwgold, Field* dwg, */
											 f64* wgalpha_dptr, f64* dwgalpha_dptr,
											 f64* F, Matrix* J) {

	const i32 NSHL = 4;
	const i32 MAX_BATCH_SIZE = 1 << 10;
	const u32 num_tet = Mesh3DNumTet(mesh);
	const u32 num_node = Mesh3DNumNode(mesh);
	const Mesh3DData* dev = Mesh3DDevice(mesh);
	const u32* ien = Mesh3DDataTet(dev);
	const f64* xg = Mesh3DDataCoord(dev);

	// f64* wgold_dptr = ArrayData(FieldDevice(wgold));
	// f64* dwgold_dptr = ArrayData(FieldDevice(dwgold));
	// f64* dwg_dptr = ArrayData(FieldDevice(dwg));



	static f64* d_shlgradu = NULL, *d_shlu = NULL, *d_gw = NULL;
	if(!d_shlgradu) {
		d_shlgradu = (f64*)CdamMallocDevice(NSHL * 3 * sizeof(f64));
		cudaMemcpy(d_shlgradu, h_shlgradu, NSHL * 3 * sizeof(f64), cudaMemcpyHostToDevice);
	}
	if(!d_shlu) {
		d_shlu = (f64*)CdamMallocDevice(NQR * NSHL * sizeof(f64));
		cudaMemcpy(d_shlu, h_shlu, NQR * NSHL * sizeof(f64), cudaMemcpyHostToDevice);
	}
	if(!d_gw) {
		d_gw = (f64*)CdamMallocDevice(NQR * sizeof(f64));
		cudaMemcpy(d_gw, h_gw, NQR * sizeof(f64), cudaMemcpyHostToDevice);
	}
	ASSERT(F || J && "Either F or J should be provided");
	printf("Assemble: %s %s\n", F ? "F" : "", J ? "J" : "");


	u32* batch_index_ptr = NULL;
	u32 max_batch_size = 0, batch_size = 0;
	cublasHandle_t handle;
	f64 one = 1.0, zero = 0.0, minus_one = -1.0;

	cublasCreate(&handle);

	f64 fact1 = kDT * kALPHAF * kGAMMA, fact2 = kDT * kALPHAF * (1.0 - kGAMMA); 
	// f64* wgalpha_dptr = (f64*)CdamMallocDevice(num_node * sizeof(f64));
	/* wgalpha_ptr[:] = wgold[:] + kDT * kALPHAF * (1.0 - kGAMMA) * dwgold[:] + kDT * kALPHAF * kGAMMA * dwg[:] */
	// cublasDcopy(handle, num_node, wgold_dptr, 1, wgalpha_dptr, 1);
	// cublasDaxpy(handle, num_node, &fact1, dwg_dptr, 1, wgalpha_dptr, 1);
	// cublasDaxpy(handle, num_node, &fact2, dwgold_dptr, 1, wgalpha_dptr, 1);

	/* Zero the RHS and LHS */
	if(F) {
		cudaMemset(F, 0, num_node * sizeof(f64));
	}
	if(J) {
		MatrixZero(J);
	}

	for(u32 b = 0; b < mesh->num_batch; ++b) {
		// batch_size = CountValueColorLegacy(mesh->color, num_tet, b);
		// ASSERT(batch_size == mesh->batch_offset[b+1] - mesh->batch_offset[b]);
		batch_size = mesh->batch_offset[b+1] - mesh->batch_offset[b];
		if (batch_size > max_batch_size) {
			max_batch_size = batch_size;
		}
	}

	// batch_index_ptr = (u32*)CdamMallocDevice(max_batch_size * sizeof(u32));

	f64* buffer = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NSHL * BS);
	f64* elem_invJ = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * (3 * 3 + 1));
	f64* shgradg = (f64*)CdamMallocDevice(max_batch_size * (sizeof(f64) * NSHL * 3 + 4 * sizeof(int)));

	f64* qr_wgalpha = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NQR * BS);
	f64* qr_dwgalpha = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NQR * BS);
	f64* qr_wggradalpha = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * 3 * BS);

	i32 num_thread = 256;
	i32 num_block = (max_batch_size + num_thread - 1) / num_thread;

	// f64* elem_buff = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NSHL * NSHL);
	f64* elem_F = NULL;
	f64* elem_J = NULL;

	if(F) {
		elem_F = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NSHL * BS);
	}
	if(J) {
		elem_J = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NSHL * NSHL * BS * BS);
	}

	f64 time_len[6] = {0, 0, 0, 0, 0, 0};
	std::chrono::steady_clock::time_point start, end;

	/* Assume all 2D arraies are column-majored */
	for(u32 b = 0; b < mesh->num_batch; ++b) {

		// batch_size = CountValueColorLegacy(mesh->color, num_tet, b);
		batch_size = mesh->batch_offset[b+1] - mesh->batch_offset[b];
		// batch_size = CountValueColor(mesh->color, num_tet, b, count_value_buffer);

		if(batch_size == 0) {
			break;
		}


		// FindValueColor(mesh->color, num_tet, b, batch_index_ptr);
		// cudaMemcpy(batch_index_ptr, mesh->batch_ind + mesh->batch_offset[b], batch_size * sizeof(u32), cudaMemcpyDeviceToDevice);
		batch_index_ptr = mesh->batch_ind + mesh->batch_offset[b];
		// CUGUARD(cudaGetLastError());
		/* 0. Calculate the element metrics */
		/* 0.0. Get dxi/dx and det(dxi/dx) */

		start = std::chrono::steady_clock::now();
		GetElemInvJ3D<u32, f64, NSHL>(batch_size, ien, batch_index_ptr, xg, elem_invJ, shgradg, handle);
		end = std::chrono::steady_clock::now();
		time_len[0] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		/* 0.1. Calculate the gradient of the shape functions */
		start = std::chrono::steady_clock::now();
		GetShapeGradKernel<u32, f64><<<CEIL_DIV(batch_size, num_thread), num_thread>>>(batch_size, elem_invJ, shgradg);
		/* 0.2. Calculate the metric of the elements */
		/* shgradg[0:3, 0:NSHL, e] = elem_invJ[0:3, 0:3, e].T @ d_shlgradu[0:3, 0:NSHL, e] for e in range(batch_size) */
		// cublasDgemmStridedBatched(handle,
		// 													CUBLAS_OP_T, CUBLAS_OP_N,
		// 													3, NSHL, 3,
		// 													&one,
		// 													elem_invJ, 3, 10ll,
		// 													d_shlgradu, 3, 0ll,
		// 													&zero,
		// 													shgradg, 3, (long long)(NSHL * 3),
		// 													batch_size);
		// CUGUARD(cudaGetLastError());
		/* 1. Interpolate the field values */  
		// cudaMemset(qr_wgalpha, 0, max_batch_size * sizeof(f64));
		// cudaMemset(qr_wggradalpha, 0, max_batch_size * sizeof(f64) * 3);
		// CUGUARD(cudaGetLastError());
		end = std::chrono::steady_clock::now();
		time_len[1] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	
		start = std::chrono::steady_clock::now();
	
		/* 1.0. Load the field value on the vertices */
		/* Load velocity */
		LoadElementValueKernel<u32, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 3,
																																																	 batch_index_ptr, ien,
																																																	 wgalpha_dptr, 3,
																																																	 buffer, BS * NSHL);
		/* Load pressure */
		LoadElementValueKernel<u32, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 1,
																																																		batch_index_ptr, ien,
																																																		dwgalpha_dptr + num_node * 3, 1,
																																																		buffer + 3, BS * NSHL);
		/* Load phi */
		LoadElementValueKernel<u32, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 1,
																																																	 batch_index_ptr, ien,
																																																	 wgalpha_dptr + num_node * 4, 1,
																																																	 buffer + 4, BS * NSHL);
		/* Load temperature */
		LoadElementValueKernel<u32, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 1,
																																																	 batch_index_ptr, ien,
																																																	 wgalpha_dptr + num_node * 5, 1,
																																																	 buffer + 5, BS * NSHL);
		// thrust::for_each(thrust::device,
		// 								 thrust::make_counting_iterator<u32>(0),
		// 								 thrust::make_counting_iterator<u32>(batch_size),
		// 								 LoadBatchFunctor<u32, f64, NSHL, 1>(batch_index_ptr, ien, wgalpha_dptr, buffer));
											
		// CUGUARD(cudaGetLastError());
		/* 1.1. Calculate the gradient*/
		/* qr_wggradalpha[0:3, 0:BS, 0:batch_size] = sum(shgradg[0:3, 0:NSHL, 0:batch_size] * buffer[0:NSHL, 0:BS, 0:batch_size], axis=1) */
		cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
															3, NSHL, BS,
															&one,
															shgradg, 3, (long long)(NSHL * 3),
															buffer, BS, (long long)(NSHL * BS),
															&zero,
															qr_wggradalpha, 3, (long long)(BS * 3),
															batch_size);
		// cublasDgemvStridedBatched(handle, CUBLAS_OP_N,
		// 													3, NSHL,
		// 													&one,
		// 													shgradg, 3, (long long)(NSHL * 3),
		// 													buffer, 1, (long long)NSHL,
		// 													&zero,
		// 													qr_wggradalpha, 1, 3ll,
		// 													batch_size);
		// CUGUARD(cudaGetLastError());

		/* 1.2. Calculate the field value on the quadrature */
		/* qr_wgalpha[0:NQR, 0:BS, 0:batch_size] = d_shlu[0:NQR, 0:NSHL, None] @ buffer[0:NSHL, 0:BS, 0:batch_size] */
		cublasDgemm(handle,
								CUBLAS_OP_N, CUBLAS_OP_N,
								NQR, batch_size * BS, NSHL,
								&one,
								d_shlu, NQR,
								buffer, NSHL,
								&zero,
								qr_wgalpha, NQR);

		// CUGUARD(cudaGetLastError());
		end = std::chrono::steady_clock::now();
		time_len[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		/* 1.3. Calculate the field value of qr_dwgalpha */
		start = std::chrono::steady_clock::now();
		// cudaMemset(qr_dwgalpha, 0, max_batch_size * sizeof(f64));
		LoadElementValueKernel<u32, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 3,
																																																	 batch_index_ptr, ien,
																																																	 dwgalpha_dptr, 3,
																																																	 buffer, BS * NSHL);
		LoadElementValueKernel<u32, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 1,
																																																	 batch_index_ptr, ien,
																																																	 dwgalpha_dptr + num_node * 3, 1,
																																																	 buffer + 3, BS * NSHL);
		LoadElementValueKernel<u32, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 1,
																																																	 batch_index_ptr, ien,
																																																	 dwgalpha_dptr + num_node * 4, 1,
																																																	 buffer + 4, BS * NSHL);
		LoadElementValueKernel<u32, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 1,
																																																	 batch_index_ptr, ien,
																																																	 dwgalpha_dptr + num_node * 5, 1,
																																																	 buffer + 5, BS * NSHL);
		// thrust::for_each(thrust::device,
		// 								 thrust::make_counting_iterator<u32>(0),
		// 								 thrust::make_counting_iterator<u32>(batch_size),
		// 								 LoadBatchFunctor<u32, f64, NSHL, 1>(batch_index_ptr, ien, dwgalpha_dptr, buffer));
		// CUGUARD(cudaGetLastError());
		/* qr_dwgalpha[0:NQR, 0:BS, 0:batch_size] = d_shlu[0:NQR, 0:NSHL] @ buffer[0:NSHL, 0:BS, 0:batch_size] */
		cublasDgemm(handle,
								CUBLAS_OP_N, CUBLAS_OP_N,
								NQR, batch_size * BS, NSHL,
								&one,
								d_shlu, NQR,
								buffer, NSHL,
								&zero,
								qr_dwgalpha,
								NQR);
		end = std::chrono::steady_clock::now();
		// CUGUARD(cudaGetLastError());
		time_len[3] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		/* 2. Calculate the elementwise residual and jacobian */
		start = std::chrono::steady_clock::now();
		IntElemAssembly<u32, f64>(batch_size,
															elem_invJ, shgradg,
															qr_wgalpha, qr_dwgalpha, qr_wggradalpha,
															elem_F, elem_J);
		end = std::chrono::steady_clock::now();
		time_len[4] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		/* 3. Assemble the global residual vector */
		start = std::chrono::steady_clock::now();
		if(F) {
			ElemRHSLocal2Global<u32, f64>(batch_size, batch_index_ptr, ien,
																		3, 
																		elem_F, BS,
																		F, 3);
			ElemRHSLocal2Global<u32, f64>(batch_size, batch_index_ptr, ien,
																		1, 
																		elem_F + 3, BS,
																		F + num_node * 3, 1);
			ElemRHSLocal2Global<u32, f64>(batch_size, batch_index_ptr, ien,
																		1, 
																		elem_F + 4, BS,
																		F + num_node * 4, 1);
			ElemRHSLocal2Global<u32, f64>(batch_size, batch_index_ptr, ien,
																		1, 
																		elem_F + 5, BS,
																		F + num_node * 5, 1);
		}
		/* 4. Assemble the global residual matrix */
		if(J) {
			// MatrixAddElementLHS(J, NSHL, BS, batch_size, ien, batch_index_ptr, elem_J, BS * NSHL);
			ElemLHSLocal2Global<u32, f64>(batch_size, batch_index_ptr, ien,
																		3, 
																		elem_J, BS * NSHL,
																		J, 3);
		}
		// cudaStreamSynchronize(0);
		end = std::chrono::steady_clock::now();
		time_len[5] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	}
	printf("Time[0]: GetElemInvJ3D: %10.4f ms\n", time_len[0]* 1e-6);
	printf("Time[1]: GetShapeGrad: %10.4f ms\n", time_len[1]* 1e-6);
	printf("Time[2]: Interpolate wg: %10.4f ms\n", time_len[2]* 1e-6);
	printf("Time[3]: Interpolate dwg: %10.4f ms\n", time_len[3]* 1e-6);
	printf("Time[4]: AssembleWeakForm: %10.4f ms\n", time_len[4]* 1e-6);
	printf("Time[5]: AssembleGlobal: %10.4f ms\n", time_len[5]* 1e-6);

	CdamFreeDevice(buffer, max_batch_size * sizeof(f64) * NSHL * BS);
	CdamFreeDevice(elem_invJ, max_batch_size * sizeof(f64) * 10);
	CdamFreeDevice(shgradg, max_batch_size * (sizeof(f64) * NSHL * 3 + 4 * sizeof(int)));


	CdamFreeDevice(elem_F, max_batch_size * sizeof(f64) * NSHL * BS);
	CdamFreeDevice(elem_J, max_batch_size * sizeof(f64) * NSHL * NSHL * BS * BS);

	CdamFreeDevice(qr_wgalpha, max_batch_size * sizeof(f64) * NQR * BS);
	CdamFreeDevice(qr_dwgalpha, max_batch_size * sizeof(f64) * NQR * BS);
	CdamFreeDevice(qr_wggradalpha, max_batch_size * sizeof(f64) * 3 * BS);
	// CdamFreeDevice(batch_index_ptr, max_batch_size * sizeof(u32));
	cublasDestroy(handle);
}

__END_DECLS__
