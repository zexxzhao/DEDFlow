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
#define kDT (5e-2)
#define kALPHAM (0.5 * (3.0 - kRHOC) / (1.0 + kRHOC))
#define kALPHAF (1.0 / (1.0 + kRHOC))
#define kGAMMA (0.5 + kALPHAM - kALPHAF)

#define NQR (4)
#define BS (6)
#define M2D(aa, ii) ((aa) * BS + (ii))
#define M4D(aa, bb, ii, jj) \
	(((aa) * (NSHL) + bb) * (BS * BS) + (ii) * BS + (jj))

#define kRHO (1.0e3)
/// #define kCP (4.2e3)
#define kCP (1.0)
#define kKAPPA (0.66)
/// #define kKAPPA (0.0)
#define kMU (10.0/3.0)

__constant__ f64 fb[3] = {0.0, 0.0, -9.81 * 0.0};
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

#define NQRB (3)
const f64 h_gwb[3] = {0.1666666666666667, 0.1666666666666667, 0.1666666666666667};
/* h_shlub[0:NSHL, 0:NQRB, 0:NFACE] */
const f64 h_shlub[NQRB*4*4] = {/* h_shlub[:, :, 0] = */
															 0.0, 0.1666666666666667, 0.1666666666666667, 0.6666666666666667,
															 0.0, 0.1666666666666667, 0.6666666666666667, 0.1666666666666667,
															 0.0, 0.6666666666666667, 0.1666666666666667, 0.1666666666666667,
															 /* h_shlub[:, :, 1] = */
															 0.1666666666666667, 0.0, 0.1666666666666667, 0.6666666666666667,
															 0.1666666666666667, 0.0, 0.6666666666666667, 0.1666666666666667,
															 0.6666666666666667, 0.0, 0.1666666666666667, 0.1666666666666667,
															 /* h_shlub[:, :, 2] = */
															 0.6666666666666667, 0.1666666666666667, 0.0, 0.1666666666666667,
															 0.1666666666666667, 0.6666666666666667, 0.0, 0.1666666666666667,
															 0.1666666666666667, 0.1666666666666667, 0.0, 0.6666666666666667,
															 /* h_shlub[:, :, 3] = */
															 0.1666666666666667, 0.6666666666666667, 0.1666666666666667, 0.0, 
															 0.1666666666666667, 0.1666666666666667, 0.6666666666666667, 0.0, 
															 0.6666666666666667, 0.1666666666666667, 0.1666666666666667, 0.0};


__constant__ f64 c_gwb[3] = {0.1666666666666667, 0.1666666666666667, 0.1666666666666667};
__constant__ f64 c_shlub[4*NQRB*4] = {/* c_shlub[:, :, 0] = */
															 0.0, 0.1666666666666667, 0.1666666666666667, 0.6666666666666667,
															 0.0, 0.1666666666666667, 0.6666666666666667, 0.1666666666666667,
															 0.0, 0.6666666666666667, 0.1666666666666667, 0.1666666666666667,
															 /* c_shlub[:, :, 1] = */
															 0.1666666666666667, 0.0, 0.1666666666666667, 0.6666666666666667,
															 0.1666666666666667, 0.0, 0.6666666666666667, 0.1666666666666667,
															 0.6666666666666667, 0.0, 0.1666666666666667, 0.1666666666666667,
															 /* c_shlub[:, :, 2] = */
															 0.6666666666666667, 0.1666666666666667, 0.0, 0.1666666666666667,
															 0.1666666666666667, 0.6666666666666667, 0.0, 0.1666666666666667,
															 0.1666666666666667, 0.1666666666666667, 0.0, 0.6666666666666667,
															 /* c_shlub[:, :, 3] = */
															 0.1666666666666667, 0.6666666666666667, 0.1666666666666667, 0.0, 
															 0.1666666666666667, 0.1666666666666667, 0.6666666666666667, 0.0, 
															 0.6666666666666667, 0.1666666666666667, 0.1666666666666667, 0.0};

__constant__ f64 c_nv[3*2*4] = {/* n[:, 0, 0] = */ -1.0, 1.0, 0.0,
															  /* n[:, 1, 0] = */ -1.0, 0.0, 1.0,
															  /* n[:, 0, 1] = */ 0.0, 0.0, 1.0,
															  /* n[:, 1, 1] = */ 0.0, 1.0, 0.0,
															  /* n[:, 0, 2] = */ 1.0, 0.0, 0.0,
															  /* n[:, 1, 2] = */ 0.0, 0.0, 1.0,
															  /* n[:, 0, 3] = */ 0.0, 1.0, 0.0,
															  /* n[:, 1, 3] = */ 1.0, 0.0, 0.0};
/* Facet normal vectors of tet in the reference coordinate*/
/* Reserved for the normal vectors in the physical coordinate using Nanson's formula */
__constant__ f64 c_nv2[4*3] = { /* 0.57735026919, 0.57735026919, 0.57735026919, */
																1.0, 1.0, 1.0,
															 -1.0, 0.0, 0.0,
																0.0, -1.0, 0.0,
																0.0, 0.0, -1.0};

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
													T* F, I ldb, const I* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batch_size * NSHL) {
		return;
	}
	if(mask && mask[i / NSHL] == 0) {
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
												 T* F, I ldb,
												 const I* mask) {
	int block_dim = 256;
	int grid_dim = CEIL_DIV(batch_size * NSHL, block_dim);
	ElemRHSLocal2GlobalKernel<I, T><<<grid_dim, block_dim>>>(batch_size, batch_index_ptr, ien,
																													 width,
																													 elem_F, lda,
																													 F, ldb, mask);
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
															  const T* elem_J,
															  Matrix* J, const I* mask) {
	int block_dim = 256;
	int grid_dim = CEIL_DIV(batch_size * NSHL * NSHL, block_dim);

	MatrixAddElemValueBlockedBatched(J, NSHL,
																	 batch_size, batch_index_ptr, ien,
																	 block_row, block_col,
																	 elem_J, block_col, block_row * block_col,
																	 mask);

}

template <typename T>
__device__ void cross_product(const T* a, const T* b, T* c) {
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}
template<typename I, typename T, I NSHL=4>
__global__ void GetElemFaceNVKernel(I batch_size, const T* metric, const I* forn, T* nv) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= batch_size) return;
	I iorn = forn[idx];
	
	nv += 3 * idx;
	/* metric (= metric + 10 * idx) is 3x3 and column-majored */
	metric += 10 * idx;
#if 0
	T a[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	/* a  is 3x2 and column-majored */
	/* d_nv (= c_nv + iorn * 6) is 3x2 and column-majored */
	const f64* d_nv = c_nv + iorn * 6;
	/* a = metric @ d_nv */

	for(I k = 0; k < 3; ++k) {
		for(I n = 0; n < 2; ++n) {
			for(I m = 0; m < 3; ++m) {
				a[n * 3 + m] += metric[k * 3 + m] * d_nv[n * 3 + k];
			}
		}
	}

	/* nv = cross(a[:, 0], a[:, 1]) */
	cross_product<T>(a, a + 3, nv);
#else
	T b[3] = {0.0, 0.0, 0.0};
	T detJ = metric[9];
	/* Using Nanson's formula */
	/* nv[0:3] = metric[0:3, 0:3] @ c_nv2[iorn, 0:3] */	
	for(I k = 0; k < 3; ++k) {
		for(I n = 0; n < 3; ++n) {
			b[n] += metric[n * 3 + k] * c_nv2[iorn * 3 + k];
		}
	}
	nv[0] = b[0] * detJ;
	nv[1] = b[1] * detJ;
	nv[2] = b[2] * detJ;
#endif
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
__device__ void GetStabTau(const T* elem_G, const T* uadv, T rho, T cp, T mu, T kappa, T dt, T* tau) {
	const T* Ginv = elem_G;
	T tau_tmp[3] = {0.0, 0.0, 0.0};

	// /* Ginv = elem_invJ[0:3, 0:3] @ elem_invJ[0:3, 0:3].T */
	// #pragma unroll
	// for(I i = 0; i < 3; ++i) {
	// 	#pragma unroll
	// 	for(I j = 0; j < 3; ++j) {
	// 		#pragma unroll
	// 		for(I k = 0; k < 3; ++k) {
	// 			Ginv[i * 3 + j] += elem_invJ[i * 3 + k] * elem_invJ[j * 3 + k];
	// 		}
	// 	}
	// }

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
	tau[0] = rsqrt(tau_tmp[0] + tau_tmp[1] + 3.0 * mu * mu * tau_tmp[2])/ rho;
	/* tauC */
	tau[1] = sqrt(tau_tmp[1] + 3.0 * mu * mu * tau_tmp[2]) / (Ginv[0] + Ginv[4] + Ginv[8]);
	/* tauPhi */
	tau[2] = rsqrt(tau_tmp[0] + tau_tmp[1]);
	/* tauT */
	tau[3] = rsqrt(tau_tmp[0] + tau_tmp[1] + 3.0 * kappa * kappa * tau_tmp[2]) / (rho * cp);
}

template<typename I, typename T, I NSHL=4>
__device__ void GetrLi(const T* du, const T* uadv, const T* gradu, const T* gradp, T rho, T mu, T* rLi) {
	#pragma unroll
	for(I i = 0; i < 3; ++i) {
		rLi[i] = du[i] - fb[i] + uadv[0] * gradu[i*3+0] + uadv[1] * gradu[i*3+1] + uadv[2] * gradu[i*3+2];
		rLi[i] = rho * rLi[i] + gradp[i];
	}
}

template<typename I, typename T, I NSHL=4, I NUM_THREAD=256>
__global__ void
AssembleWeakFormLHSKernel(I batch_size,
													const T* elem_G, const T* shgradg,
													const T* qr_wgalpha, const T* qr_dwgalpha, const T* qr_wggradalpha,
													T* elem_J) {
	I idx = blockIdx.x * blockDim.x + threadIdx.x;
	I tx = threadIdx.x;
	I te, ti, lane;
	I aa = (idx % (NSHL * NSHL)) / NSHL;
	I bb = (idx % (NSHL * NSHL)) % NSHL;

	I iel = idx / (NSHL * NSHL);


	constexpr I BLK_ELEM_COUNT_MAX = NUM_THREAD / (NSHL * NSHL);
	I BLK_ELEM_BEGIN = blockIdx.x * blockDim.x / (NSHL * NSHL);
	I BLK_ELEM_COUNT = (BLK_ELEM_COUNT_MAX < batch_size - BLK_ELEM_BEGIN ? BLK_ELEM_COUNT_MAX : batch_size - BLK_ELEM_BEGIN);

	const f64 fact1 = kALPHAM;
	const f64 fact2 = kDT * kALPHAF * kGAMMA;

	__shared__ T sh_buff[BLK_ELEM_COUNT_MAX * (NSHL * NSHL + 4)];
	typedef T (T12)[NSHL * 3];
	typedef T (T4)[NSHL];
	T12* shgradl = (T12*)sh_buff;
	T4* shconv = (T4*)(sh_buff + BLK_ELEM_COUNT_MAX * NSHL * 3);
	T4* tau_component = (T4*)(sh_buff + BLK_ELEM_COUNT_MAX * NSHL * NSHL);

	if(tx < BLK_ELEM_COUNT * NSHL * 3) {
		((T*)shgradl)[tx] = shgradg[BLK_ELEM_BEGIN * NSHL * 3 + tx];
	}
	if(tx < BLK_ELEM_COUNT) {
		T gg = 0.0, tr = 0.0, gij;
		for(I i = 0; i < 9; ++i) {
			gij = elem_G[(BLK_ELEM_BEGIN + tx) * 10 + i];
			gg += gij * gij;
			tr += gij * !(i & 0x3);
		}
		tau_component[tx][0] = gg;
		tau_component[tx][1] = 1.0 / tr;
	}
	__syncthreads();



	/* Column-majored */
	/* [:, 0:3]: Velocity */
	/* [:, 3]: Pressure */
	/* [:, 4]: Phi */
	/* [:, 5]: Temperature */
	// qr_wgalpha += iel * NQR * BS; /* (NQR, BS) */
	// qr_dwgalpha += iel * NQR * BS; /* (NQR, BS) */
	// qr_wggradalpha += iel * 3 * BS; /* (3, BS) */


	// for(I i = 0; i < NSHL * NSHL * BS * BS; ++i) {
	// 	elem_J[i] = 0.0;
	// }

	T tau[4] = {0.0, 0.0, 0.0, 0.0}; 
	// T divu = qr_wggradalpha[iel * 3 * BS + 0]
	// 				+ qr_wggradalpha[iel * 3 * BS + 4]
	// 				+ qr_wggradalpha[iel * 3 * BS + 8];
	
	// T* elem_J_buffer = elem_J + (iel * NSHL * NSHL + aa * NSHL + bb) * BS * BS;
	if(iel >= batch_size) return;

	T elem_J_buffer[4*4];
	for(I i = 0; i < 4*4; ++i) {
		elem_J_buffer[i] = 0.0;
	}

	T tmp;
	T detJ = elem_G[iel * 10 + 9];
	const T knu = kMU / kRHO;
	const T kalpha = kKAPPA / kRHO / kCP;
	for(I iq = 0; iq < NQR; ++iq) {
		/* Update shconv[:][:] */
		if(tx < BLK_ELEM_COUNT * NSHL) {
			te = tx / NSHL;
			lane = tx % NSHL;
			/* tx = te * NSHL + lane */
			/* shconv[te][lane] = shgradl[te][lane][0:3] * uadv[0:3] */
			shconv[te][lane] = 0.0;
			shconv[te][lane] += shgradl[te][lane * 3 + 0] * qr_wgalpha[NQR * BS * (BLK_ELEM_BEGIN + te) + 0 * NQR + iq];
			shconv[te][lane] += shgradl[te][lane * 3 + 1] * qr_wgalpha[NQR * BS * (BLK_ELEM_BEGIN + te) + 1 * NQR + iq];
			shconv[te][lane] += shgradl[te][lane * 3 + 2] * qr_wgalpha[NQR * BS * (BLK_ELEM_BEGIN + te) + 2 * NQR + iq];
		}
		__syncthreads();
		/* Get tau */
		/* Update tau_component[:][0] */
		if(tx < BLK_ELEM_COUNT) {
			// T uadv[3];
			// uadv[0] = qr_wgalpha[NQR * BS * (BLK_ELEM_BEGIN + tx) + 0 * NQR + iq];
			// uadv[1] = qr_wgalpha[NQR * BS * (BLK_ELEM_BEGIN + tx) + 1 * NQR + iq];
			// uadv[2] = qr_wgalpha[NQR * BS * (BLK_ELEM_BEGIN + tx) + 2 * NQR + iq];
			tmp = 0;
			tmp += shconv[tx][1] * shconv[tx][1];
			tmp += shconv[tx][2] * shconv[tx][2];
			tmp += shconv[tx][3] * shconv[tx][3];
			// for(I i = 0; i < 3; ++i) {
			// 	for(I j = 0; j < 3; ++j) {
			// 		tmp += uadv[i] * elem_G[(BLK_ELEM_BEGIN + tx) * 10 + i * 3 + j] * uadv[j];
			// 	}
			// }
			tau_component[tx][2] = rsqrt(4.0 / (kDT * kDT) + tmp + 3.0 * knu * knu * tau_component[tx][0]) / kRHO;
			tau_component[tx][3] = sqrt(tmp + 3.0 * knu * knu * tau_component[tx][0]) * tau_component[tx][1];
		}
		__syncthreads();

		te = tx / (NSHL * NSHL);

		tau[0] = tau_component[te][2];
		tau[1] = tau_component[te][3];
		// tau[0] = 0.0;
		// tau[1] = 0.0;
		// tau[2] = rsqrt(4.0 / (kDT * kDT) + tau_component[te]);
		// tau[3] = rsqrt(4.0 / (kDT * kDT) + tau_component[te] + 3.0 * kalpha * kalpha * tau_component[te + BLK_ELEM_COUNT]) / (kRHO * kCP);

		T eK = shgradl[te][aa * 3 + 0] * shgradl[te][bb * 3 + 0] +
						shgradl[te][aa * 3 + 1] * shgradl[te][bb * 3 + 1] +
						shgradl[te][aa * 3 + 2] * shgradl[te][bb * 3 + 2];
		T detJgw = detJ * gw[iq];
		tmp = 0.0;
		tmp += fact1 * kRHO * shlu[aa * NQR + iq] * shlu[bb * NQR + iq];
		tmp += fact1 * kRHO * kRHO * tau[0] * shconv[te][aa] * shlu[bb * NQR + iq];
		tmp += fact2 * shlu[aa * NQR + iq] * kRHO * shconv[te][bb];
		tmp += fact2 * tau[0] * kRHO * shconv[te][aa] * kRHO * shconv[te][bb];
		tmp += fact2 * kMU * eK;
		// tmp += fact2 * kMU * (shgradl[te][aa * 3 + 0] * shgradl[te][bb * 3 + 0] +
		// 											shgradl[te][aa * 3 + 1] * shgradl[te][bb * 3 + 1] +
		// 											shgradl[te][aa * 3 + 2] * shgradl[te][bb * 3 + 2]);
		/* dRM/dU */
		// elem_J[M4D(aa, bb, 0, 0)] += tmp[aa][bb] * gw[iq] * detJ;
		// elem_J[M4D(aa, bb, 1, 1)] += tmp[aa][bb] * gw[iq] * detJ;
		// elem_J[M4D(aa, bb, 2, 2)] += tmp[aa][bb] * gw[iq] * detJ;
		// elem_J[M4D(aa, bb, 0, 0)] += tmp * gw[iq] * detJ;
		// elem_J[M4D(aa, bb, 1, 1)] += tmp * gw[iq] * detJ;
		// elem_J[M4D(aa, bb, 2, 2)] += tmp * gw[iq] * detJ;
		elem_J_buffer[0 * 4 + 0] += tmp * detJgw;
		elem_J_buffer[1 * 4 + 1] += tmp * detJgw;
		elem_J_buffer[2 * 4 + 2] += tmp * detJgw;
		for(I ii = 0; ii < 3; ++ii) {
			for(I jj = 0; jj < 3; ++jj) {
				elem_J_buffer[ii * 4 + jj] += fact2 * kMU * shgradl[te][aa * 3 + jj] * shgradl[te][bb * 3 + ii] * detJgw;
				elem_J_buffer[ii * 4 + jj] += fact2 * kRHO * tau[1] * shgradl[te][aa * 3 + ii] * shgradl[te][bb * 3 + jj] * detJgw;
			}
		}

		/* dRM/dP */
		for(I ii = 0; ii < 3; ++ii) {
			elem_J_buffer[ii * 4 + 3] -= shgradl[te][aa * 3 + ii] * shlu[bb * NQR + iq] * detJgw;
			elem_J_buffer[ii * 4 + 3] += kRHO * tau[0] * shconv[te][aa] * shgradl[te][bb * 3 + ii] * detJgw;
		}

		// tau[0] = 0.0;
		/* dRC/dU */
		for(I ii = 0; ii < 3; ++ii) {
			elem_J_buffer[3 * 4 + ii] += fact1 * kRHO * tau[0] * shgradl[te][aa * 3 + ii] * shlu[bb * NQR + iq] * detJgw;
			elem_J_buffer[3 * 4 + ii] += fact2 * shlu[aa * NQR + iq] * shgradl[te][bb * 3 + ii] * detJgw;
			elem_J_buffer[3 * 4 + ii] += fact2 * tau[0] * shgradl[te][aa * 3 + ii] * kRHO * shconv[te][bb] * detJgw;
		}

		/* dRC/dP */
		// elem_J_buffer[M2D(3, 3)] += tau[0] * eK * gw[iq] * detJ;
		elem_J_buffer[3 * 4 + 3] += tau[0] * eK * detJgw;
		// elem_J_buffer[M2D(3, 3)] += tau[0] * (shgradl[te][aa * 3 + 0] * shgradl[te][bb * 3 + 0] +
		// 																			shgradl[te][aa * 3 + 1] * shgradl[te][bb * 3 + 1] +
		// 																			shgradl[te][aa * 3 + 2] * shgradl[te][bb * 3 + 2]) * gw[iq] * detJ;

		/* dRphi/dphi */
		// elem_J[M4D(aa, bb, 4, 4)] += (shlu[aa * NQR + iq] + tau[2] * shconv[aa]) 
		// 														* (fact1 * shlu[bb * NQR + iq] + fact2 * shconv[bb]) * gw[iq] * detJ;
		// elem_J_buffer[M2D(4, 4)] += aa == bb;

		/* dRT/dT */
		// elem_J[M4D(aa, bb, 5, 5)] += (shlu[aa * NQR + iq] + kRHO * kCP * tau[3] * shconv[aa]) 
		// 														* kRHO * kCP * (fact1 * shlu[bb * NQR + iq] + fact2 * shconv[bb]) * gw[iq] * detJ;
		// elem_J[M4D(aa, bb, 5, 5)] += kKAPPA * (shgradl[aa * 3 + 0] * shgradl[bb * 3 + 0] +
		// 																			 shgradl[aa * 3 + 1] * shgradl[bb * 3 + 1] +
		// 																			 shgradl[aa * 3 + 2] * shgradl[bb * 3 + 2]) * gw[iq] * detJ;
		// elem_J_buffer[M2D(5, 5)] += aa == bb;
		// for(int ii = 0; ii < BS; ++ii) {
		// 	for(int jj = 0; jj < BS; ++jj) {
		// 		elem_J[M4D(aa, bb, ii, jj)] = ii * BS + jj + 1.0;
		// 	}
		// }
#ifdef DBG_TET
		if(batch_size > 0 && aa == 0 && bb == 0 && iel == 0 && iq >= 0) {
			printf("SHARED MEM[%d][%d][%d]\n", iel, aa, bb);
			printf("tau: %.17e %.17e %.17e %.17e\n", tau[0], tau[1], tau[2], tau[3]);
			printf("tmp: %.17e detJ: %.17e\n", tmp, detJ);
			printf("tau_comp: %.17e %.17e\n", tau_component[te][0], tau_component[te][1]);
			printf("shgradl[0:3]: %.17e %.17e %.17e\n", shgradl[te][0], shgradl[te][1], shgradl[te][2]);
			printf("shgradl[3:6]: %.17e %.17e %.17e\n", shgradl[te][3], shgradl[te][4], shgradl[te][5]);
			printf("shgradl[6:9]: %.17e %.17e %.17e\n", shgradl[te][6], shgradl[te][7], shgradl[te][8]);
			printf("shgradl[9:12]: %.17e %.17e %.17e\n", shgradl[te][9], shgradl[te][10], shgradl[te][11]);
			printf("shconv[0:4]: %.17e %.17e %.17e %.17e\n", shconv[te][0], shconv[te][1], shconv[te][2], shconv[te][3]);

			if(iq == 3) {
				printf("%.17e %.17e %.17e %.17e\n", elem_J_buffer[0], elem_J_buffer[1], elem_J_buffer[2], elem_J_buffer[3]);
				printf("%.17e %.17e %.17e %.17e\n", elem_J_buffer[4], elem_J_buffer[5], elem_J_buffer[6], elem_J_buffer[7]);
				printf("%.17e %.17e %.17e %.17e\n", elem_J_buffer[8], elem_J_buffer[9], elem_J_buffer[10], elem_J_buffer[11]);
				printf("%.17e %.17e %.17e %.17e\n", elem_J_buffer[12], elem_J_buffer[13], elem_J_buffer[14], elem_J_buffer[15]);
			}

			// if(iq == 3) {
			// 	printf("elem_J[%d][%d][%d]: %.17e %.17e %.17e %.17e %.17e %.17e\n", iel, aa, bb, elem_J_buffer[0], elem_J_buffer[1], elem_J_buffer[2], elem_J_buffer[3], elem_J_buffer[4], elem_J_buffer[5]);
			// 	printf("elem_J[%d][%d][%d]: %.17e %.17e %.17e %.17e %.17e %.17e\n", iel, aa, bb, elem_J_buffer[6], elem_J_buffer[7], elem_J_buffer[8], elem_J_buffer[9], elem_J_buffer[10], elem_J_buffer[11]);
			// 	printf("elem_J[%d][%d][%d]: %.17e %.17e %.17e %.17e %.17e %.17e\n", iel, aa, bb, elem_J_buffer[12], elem_J_buffer[13], elem_J_buffer[14], elem_J_buffer[15], elem_J_buffer[16], elem_J_buffer[17]);
			// 	printf("elem_J[%d][%d][%d]: %.17e %.17e %.17e %.17e %.17e %.17e\n", iel, aa, bb, elem_J_buffer[18], elem_J_buffer[19], elem_J_buffer[20], elem_J_buffer[21], elem_J_buffer[22], elem_J_buffer[23]);
			// 	printf("elem_J[%d][%d][%d]: %.17e %.17e %.17e %.17e %.17e %.17e\n", iel, aa, bb, elem_J_buffer[24], elem_J_buffer[25], elem_J_buffer[26], elem_J_buffer[27], elem_J_buffer[28], elem_J_buffer[29]);
			// 	printf("elem_J[%d][%d][%d]: %.17e %.17e %.17e %.17e %.17e %.17e\n", iel, aa, bb, elem_J_buffer[30], elem_J_buffer[31], elem_J_buffer[32], elem_J_buffer[33], elem_J_buffer[34], elem_J_buffer[35]);
			// }
		}
#endif
		__syncthreads();
	}

	if constexpr(0){
		T* buff = sh_buff;
		buff[tx] = 0.0;

		I THREAD_STRIDE = 8; /* Hard coded for 256 threads and 24 leading dimensions */
		I wrap_id = tx / 32;
		lane = tx % 32;
		for(I e = 0; e < BLK_ELEM_COUNT * 2; ++e) {
			/* Load elem_J_buffer into buff */
			/* The thread range is [e * THREAD_STRIDE, (e + 1) * THREAD_STRIDE) */ 
			I tx_start = e * THREAD_STRIDE;
			I tx_end = tx_start + THREAD_STRIDE;
			if(tx_start <= tx && tx < tx_end) {
				#pragma unroll
				for(I i = 0; i < 4; i++) {
					#pragma unroll
					for(I j = 0; j < 4; j++) {
						buff[(tx - tx_start) * 24 + i * 6 + j] = elem_J_buffer[i * 4 + j];
					}
				}
			}
			__syncthreads();

			/* Update elem_J */
			I e_idx = BLK_ELEM_BEGIN + e / 2;
			T* dst = elem_J + (e_idx * NSHL * NSHL + wrap_id + (e & 1) * THREAD_STRIDE) * BS * BS;
			if(lane < 24) {
				dst[lane] = buff[tx];
			}
		}
	}
	else {
		elem_J += (iel * NSHL * NSHL + aa * NSHL + bb) * BS * BS;
		#pragma unroll
		for(I i = 0; i < 4; ++i) {
			#pragma unroll
			for(I j = 0; j < 4; ++j) {
				// elem_J[i * BS + j] += elem_J_buffer[i * 4 + j];
				atomicAdd(elem_J + i * BS + j, elem_J_buffer[i * 4 + j]);
			}
		}
	}
	elem_J[M2D(4, 4)] += aa == bb;
	elem_J[M2D(5, 5)] += aa == bb; 
}

template<typename I, typename T, I NSHL=4, I TENSOR=0>
__global__ void
AssembleWeakFormKernel(I batch_size,
											 const T* elem_G, const T* shgradg,
											 const T* qr_wgalpha, const T* qr_dwgalpha, const T* qr_wggradalpha,
											 T* elem_F, T* elem_J) {
	I iel = blockIdx.x * blockDim.x + threadIdx.x;
	if(iel >= batch_size) return;

	elem_G += iel * 10;
	T detJ = elem_G[9];
	shgradg += iel * NSHL * 3;
	/* Column-majored */
	/* [:, 0:3]: Velocity */
	/* [:, 3]: Pressure */
	/* [:, 4]: Phi */
	/* [:, 5]: Temperature */
	qr_wgalpha += iel * NQR * BS; /* (NQR, BS) */
	qr_dwgalpha += iel * NQR * BS; /* (NQR, BS) */
	qr_wggradalpha += iel * 3 * BS; /* (3, BS) */

	if constexpr(TENSOR == 1) {
		elem_F += iel * NSHL * BS;
		for(I i = 0; i < NSHL * BS; ++i) {
			elem_F[i] = 0.0;
		}
	}

	if constexpr (TENSOR == 2) {
		elem_J += iel * NSHL * NSHL * BS * BS;
		for(I i = 0; i < NSHL * NSHL * BS * BS; ++i) {
			elem_J[i] = 0.0;
		}
	}

	T tau[4] = {0.0, 0.0, 0.0, 0.0}; 
	T divu = qr_wggradalpha[0] + qr_wggradalpha[4] + qr_wggradalpha[8];
	
	T rLi[3];
	T uadv[3];
	T shconv[NSHL];

	for(I iq = 0; iq < NQR; ++iq) {
		/* Get uadv */
		uadv[0] = qr_wgalpha[NQR * 0 + iq];
		uadv[1] = qr_wgalpha[NQR * 1 + iq];
		uadv[2] = qr_wgalpha[NQR * 2 + iq];

		/* Get rLi */
		for(I i = 0; i < 3; ++i) {
			rLi[i] = 0.0;
			rLi[i] += kRHO * (qr_dwgalpha[NQR * i + iq] - fb[i]);
			rLi[i] += kRHO * uadv[0] * qr_wggradalpha[3 * i + 0];
			rLi[i] += kRHO * uadv[1] * qr_wggradalpha[3 * i + 1];
			rLi[i] += kRHO * uadv[2] * qr_wggradalpha[3 * i + 2];
			rLi[i] += qr_wggradalpha[3 * 3 + i];
		}

		/* Get Stab */
		GetStabTau<I, T>(elem_G, uadv, kRHO, kCP, kMU, kKAPPA, kDT, tau);

		for(I aa = 0; aa < NSHL; ++aa) {
			// shconv[aa] = uadv[0] * shgradg[aa * 3 + 0] + uadv[1] * shgradg[aa * 3 + 1] + uadv[2] * shgradg[aa * 3 + 2];
			shconv[aa] = 0.0;
			shconv[aa] += uadv[0] * shgradg[aa * 3 + 0];
			shconv[aa] += uadv[1] * shgradg[aa * 3 + 1];
			shconv[aa] += uadv[2] * shgradg[aa * 3 + 2];
		}

		if constexpr(TENSOR == 1) {
			// tau[0] = 0;
			// tau[1] = 0;
			T tmp0[3];
			T tmp1[3 * 3];
			// T tmp2[1];
			/* Get tmp0 */
			for(I i = 0; i < 3; ++i) {
				tmp0[i] = 0.0;
				tmp0[i] += kRHO * (qr_dwgalpha[NQR * i + iq] - fb[i]);
				tmp0[i] += kRHO * (uadv[0] - tau[0] * rLi[0]) * qr_wggradalpha[3 * i + 0];
				tmp0[i] += kRHO * (uadv[1] - tau[0] * rLi[1]) * qr_wggradalpha[3 * i + 1];
				tmp0[i] += kRHO * (uadv[2] - tau[0] * rLi[2]) * qr_wggradalpha[3 * i + 2];
			}

			/* Get tmp1 */
			for(I i = 0; i < 3; ++i) {
				for(I j = 0; j < 3; ++j) {
					tmp1[i * 3 + j] = 0.0;
					tmp1[i * 3 + j] += kMU * (qr_wggradalpha[3 * i + j] + qr_wggradalpha[3 * j + i]);
					tmp1[i * 3 + j] += kRHO * tau[0] * rLi[i] * uadv[j];
					tmp1[i * 3 + j] -= kRHO * tau[0] * tau[0] * rLi[i] * rLi[j];
				}
			}
			for(I i = 0; i < 3; ++i) {
				tmp1[i * 3 + i] += -qr_wgalpha[NQR * 3 + iq] + kRHO * tau[1] * divu;
			}

			/* Get tmp2 */
			// tmp2[0] = -qr_wgalpha[NQR * 3 + iq] + kRHO * tau[1] * divu;

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
					// bm += shgradg[aa * 3 + ii] * tmp2[0];
					elem_F[M2D(aa, ii)] += bm * gw[iq] * detJ;
				}
			}
			/* Continuity */
			#pragma unroll
			for(I aa = 0; aa < NSHL; ++aa) {
				T bc = 0.0;
				bc += shlu[aa * NQR + iq] * divu;
				bc += tau[0] * rLi[0] * shgradg[aa * 3 + 0];
				bc += tau[0] * rLi[1] * shgradg[aa * 3 + 1];
				bc += tau[0] * rLi[2] * shgradg[aa * 3 + 2];
				elem_F[M2D(aa, 3)] += bc * gw[iq] * detJ;
			}
			/* Phi */
			for(I aa = 0; aa < NSHL; ++aa) {
				T bp = qr_dwgalpha[NQR * 4 + iq]
						 + uadv[0] * qr_wggradalpha[3 * 4 + 0]
						 + uadv[1] * qr_wggradalpha[3 * 4 + 1]
						 + uadv[2] * qr_wggradalpha[3 * 4 + 2];
				elem_F[M2D(aa, 4)] += bp * (shlu[aa * NQR + iq] + tau[2] * shconv[aa]) * gw[iq] * detJ;
			}

			/* Temperature */
			for(I aa = 0; aa < NSHL; ++aa) {
				T bt = kRHO * kCP * (qr_dwgalpha[NQR * 5 + iq]
						 + uadv[0] * qr_wggradalpha[3 * 5 + 0]
						 + uadv[1] * qr_wggradalpha[3 * 5 + 1]
						 + uadv[2] * qr_wggradalpha[3 * 5 + 2]) 
						 * (shlu[aa * NQR + iq] + kRHO * kCP * tau[3] * shconv[aa]);

				bt += kKAPPA * (qr_wggradalpha[3 * 5 + 0] * shgradg[aa * 3 + 0] +
												qr_wggradalpha[3 * 5 + 1] * shgradg[aa * 3 + 1] +
												qr_wggradalpha[3 * 5 + 2] * shgradg[aa * 3 + 2]);

				elem_F[M2D(aa, 5)] += bt * gw[iq] * detJ;
			}
#ifdef DBG_TET
			if(0 && iel == 0) {
				printf("tmp0: %.17e %.17e %.17e\n", tmp0[0], tmp0[1], tmp0[2]);
				printf("tmp1[0, :] = %.17e %.17e %.17e\n", tmp1[0], tmp1[1], tmp1[2]);
				printf("tmp1[1, :] = %.17e %.17e %.17e\n", tmp1[3], tmp1[4], tmp1[5]);
				printf("tmp1[2, :] = %.17e %.17e %.17e\n", tmp1[6], tmp1[7], tmp1[8]);
				printf("gw=%.17e detJ=%.17e\n", gw[iq], detJ);
				if(iq == 3) {
					printf("elem_F(%d) = %p\n", iq, elem_F);
					printf("F[:, 0] = %.17e, %.17e, %.17e, %.17e, %.17e, %.17e\n", elem_F[0], elem_F[1], elem_F[2], elem_F[3], elem_F[4], elem_F[5]);
					printf("F[:, 1] = %.17e, %.17e, %.17e, %.17e, %.17e, %.17e\n", elem_F[6], elem_F[7], elem_F[8], elem_F[9], elem_F[10], elem_F[11]);
					printf("F[:, 2] = %.17e, %.17e, %.17e, %.17e, %.17e, %.17e\n", elem_F[12], elem_F[13], elem_F[14], elem_F[15], elem_F[16], elem_F[17]);
					printf("F[:, 3] = %.17e, %.17e, %.17e, %.17e, %.17e, %.17e\n", elem_F[18], elem_F[19], elem_F[20], elem_F[21], elem_F[22], elem_F[23]);
				}
			}
#endif
		}

		if constexpr(TENSOR == 2) {
			f64 fact1 = kALPHAM;
			f64 fact2 = kDT * kALPHAF * kGAMMA;

			f64 shgradl[NSHL*3];
			for(I aa = 0; aa < NSHL * 3; ++aa) {
				shgradl[aa] = shgradg[aa];
			}

			// T tmp[NSHL][NSHL];
			// for(I aa = 0; aa < NSHL; aa++) {
			// 	for(I bb = 0; bb < NSHL; bb++) {
			// 		tmp[aa][bb] = 0.0;
			// 		tmp[aa][bb] += fact1 * kRHO * shlu[aa * NQR + iq] * shlu[bb * NQR + iq];
			// 		tmp[aa][bb] += fact1 * kRHO * kRHO * tau[0] * shconv[aa] * shlu[bb * NQR + iq];
			// 		tmp[aa][bb] += fact2 * shlu[aa * NQR + iq] * kRHO * shconv[bb];
			// 		tmp[aa][bb] += fact2 * tau[0] * kRHO * shconv[aa] * kRHO * shconv[bb];
			// 		tmp[aa][bb] += fact2 * kMU * (shgradg[aa * 3 + 0] * shgradg[bb * 3 + 0] +
			// 																	shgradg[aa * 3 + 1] * shgradg[bb * 3 + 1] +
			// 																	shgradg[aa * 3 + 2] * shgradg[bb * 3 + 2]);
			// 	}
			// }

			T tmp;
			for(I aa = 0; aa < NSHL; ++aa) {
				for(I bb = 0; bb < NSHL; ++bb) {
					tmp = 0.0;
					tmp += fact1 * kRHO * shlu[aa * NQR + iq] * shlu[bb * NQR + iq];
					tmp += fact1 * kRHO * kRHO * tau[0] * shconv[aa] * shlu[bb * NQR + iq];
					tmp += fact2 * shlu[aa * NQR + iq] * kRHO * shconv[bb];
					tmp += fact2 * tau[0] * kRHO * shconv[aa] * kRHO * shconv[bb];
					tmp += fact2 * kMU * (shgradl[aa * 3 + 0] * shgradl[bb * 3 + 0] +
																shgradl[aa * 3 + 1] * shgradl[bb * 3 + 1] +
																shgradl[aa * 3 + 2] * shgradl[bb * 3 + 2]);
					/* dRM/dU */
					// elem_J[M4D(aa, bb, 0, 0)] += tmp[aa][bb] * gw[iq] * detJ;
					// elem_J[M4D(aa, bb, 1, 1)] += tmp[aa][bb] * gw[iq] * detJ;
					// elem_J[M4D(aa, bb, 2, 2)] += tmp[aa][bb] * gw[iq] * detJ;
					elem_J[M4D(aa, bb, 0, 0)] += tmp * gw[iq] * detJ;
					elem_J[M4D(aa, bb, 1, 1)] += tmp * gw[iq] * detJ;
					elem_J[M4D(aa, bb, 2, 2)] += tmp * gw[iq] * detJ;
					for(I ii = 0; ii < 3; ++ii) {
						for(I jj = 0; jj < 3; ++jj) {
							elem_J[M4D(aa, bb, ii, jj)] += fact2 * kMU * shgradl[aa * 3 + jj] * shgradl[bb * 3 + ii] * gw[iq] * detJ;
							elem_J[M4D(aa, bb, ii, jj)] += fact2 * kRHO * tau[1] * shgradl[aa * 3 + ii] * shgradl[bb * 3 + jj] * gw[iq] * detJ;
						}
					}

					/* dRC/dU */
					for(I ii = 0; ii < 3; ++ii) {
						elem_J[M4D(aa, bb, 3, ii)] += fact1 * kRHO * tau[0] * shgradl[aa * 3 + ii] * shlu[bb * NQR + iq] * gw[iq] * detJ;
						elem_J[M4D(aa, bb, 3, ii)] += fact2 * shlu[aa * NQR + iq] * shgradl[bb * 3 + ii] * gw[iq] * detJ;
						elem_J[M4D(aa, bb, 3, ii)] += fact2 * tau[0] * shgradl[aa * 3 + ii] * kRHO * shconv[bb] * gw[iq] * detJ;
					}

					/* dRM/dP */
					for(I ii = 0; ii < 3; ++ii) {
						elem_J[M4D(aa, bb, ii, 3)] -= shgradl[aa * 3 + ii] * shlu[bb * NQR + iq] * gw[iq] * detJ;
						elem_J[M4D(aa, bb, ii, 3)] -= kRHO * tau[0] * shconv[aa] * shgradl[bb * 3 + ii] * gw[iq] * detJ;
					}

					/* dRC/dP */
					elem_J[M4D(aa, bb, 3, 3)] += tau[0] * (shgradl[aa * 3 + 0] * shgradl[bb * 3 + 0] +
																								 shgradl[aa * 3 + 1] * shgradl[bb * 3 + 1] +
																								 shgradl[aa * 3 + 2] * shgradl[bb * 3 + 2]) * gw[iq] * detJ;

					/* dRphi/dphi */
					// elem_J[M4D(aa, bb, 4, 4)] += (shlu[aa * NQR + iq] + tau[2] * shconv[aa]) 
					// 														* (fact1 * shlu[bb * NQR + iq] + fact2 * shconv[bb]) * gw[iq] * detJ;
					elem_J[M4D(aa, bb, 4, 4)] += aa == bb;

					/* dRT/dT */
					// elem_J[M4D(aa, bb, 5, 5)] += (shlu[aa * NQR + iq] + kRHO * kCP * tau[3] * shconv[aa]) 
					// 														* kRHO * kCP * (fact1 * shlu[bb * NQR + iq] + fact2 * shconv[bb]) * gw[iq] * detJ;
					// elem_J[M4D(aa, bb, 5, 5)] += kKAPPA * (shgradl[aa * 3 + 0] * shgradl[bb * 3 + 0] +
					// 																			 shgradl[aa * 3 + 1] * shgradl[bb * 3 + 1] +
					// 																			 shgradl[aa * 3 + 2] * shgradl[bb * 3 + 2]) * gw[iq] * detJ;
					elem_J[M4D(aa, bb, 5, 5)] += aa == bb;
					// for(int ii = 0; ii < BS; ++ii) {
					// 	for(int jj = 0; jj < BS; ++jj) {
					// 		elem_J[M4D(aa, bb, ii, jj)] = ii * BS + jj + 1.0;
					// 	}
					// }
					if(0 && batch_size > 28700 && aa == 1 && bb == 2 && iel == 0 && iq >= 0) {
						printf("NAIVE[%d][%d][%d]\n", iel, aa, bb);
						printf("tau: %.17e %.17e %.17e %.17e\n", tau[0], tau[1], tau[2], tau[3]);
						printf("tmp: %.17e detJ: %.17e\n", tmp, detJ);
						printf("shgradl[0:3]: %.17e %.17e %.17e\n", shgradl[0], shgradl[1], shgradl[2]);
						printf("shgradl[3:6]: %.17e %.17e %.17e\n", shgradl[3], shgradl[4], shgradl[5]);
						printf("shgradl[6:9]: %.17e %.17e %.17e\n", shgradl[6], shgradl[7], shgradl[8]);
						printf("shgradl[9:12]: %.17e %.17e %.17e\n", shgradl[9], shgradl[10], shgradl[11]);
						printf("shconv[0:4]: %.17e %.17e %.17e %.17e\n", shconv[0], shconv[1], shconv[2], shconv[3]);
						if(iq == 3) {
							T* elem_J_buffer = elem_J + (aa * NSHL + bb) * BS * BS;
							printf("elem_J[%d][%d][%d]: %.17e %.17e %.17e %.17e %.17e %.17e\n", iel, aa, bb, elem_J_buffer[0], elem_J_buffer[1], elem_J_buffer[2], elem_J_buffer[3], elem_J_buffer[4], elem_J_buffer[5]);
							printf("elem_J[%d][%d][%d]: %.17e %.17e %.17e %.17e %.17e %.17e\n", iel, aa, bb, elem_J_buffer[6], elem_J_buffer[7], elem_J_buffer[8], elem_J_buffer[9], elem_J_buffer[10], elem_J_buffer[11]);
							printf("elem_J[%d][%d][%d]: %.17e %.17e %.17e %.17e %.17e %.17e\n", iel, aa, bb, elem_J_buffer[12], elem_J_buffer[13], elem_J_buffer[14], elem_J_buffer[15], elem_J_buffer[16], elem_J_buffer[17]);
							printf("elem_J[%d][%d][%d]: %.17e %.17e %.17e %.17e %.17e %.17e\n", iel, aa, bb, elem_J_buffer[18], elem_J_buffer[19], elem_J_buffer[20], elem_J_buffer[21], elem_J_buffer[22], elem_J_buffer[23]);
							printf("elem_J[%d][%d][%d]: %.17e %.17e %.17e %.17e %.17e %.17e\n", iel, aa, bb, elem_J_buffer[24], elem_J_buffer[25], elem_J_buffer[26], elem_J_buffer[27], elem_J_buffer[28], elem_J_buffer[29]);
							printf("elem_J[%d][%d][%d]: %.17e %.17e %.17e %.17e %.17e %.17e\n", iel, aa, bb, elem_J_buffer[30], elem_J_buffer[31], elem_J_buffer[32], elem_J_buffer[33], elem_J_buffer[34], elem_J_buffer[35]);

						}
					}

				}
			}
		}
	}
	

}

template <typename I, typename T, I NSHL=4>
__global__ void
FaceAssemblyKernel(I batch_size, const T* elem_invJ, const T* nv, const T* shgradg,
									 const I* forn, const T* qr_wgalpha, const T* qr_wggradalpha,
									 T* elem_F, T* elem_J) {
	I idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= batch_size) return;

	I iorn = forn[idx];
	elem_invJ += idx * 10;
	nv += idx * 3;
	shgradg += idx * 3 * NSHL;
	qr_wgalpha += idx * NQRB * BS;
	qr_wggradalpha += idx * 3 * BS;


	T hinv = 0.0, detJb = 0.0;
	T uadv[3];
	for(I i = 0; i < 3; ++i) {
		uadv[i] = elem_invJ[i + 3 * 0] * nv[0] + elem_invJ[i + 3 * 1] * nv[1] + elem_invJ[i + 3 * 2] * nv[2];
		hinv += uadv[i] * uadv[i];
		detJb += nv[i] * nv[i];
	}
	detJb = sqrt(detJb);
	// hinv = sqrt(hinv) / detJb;
	hinv = sqrt(hinv);
	T tau_b = 4.0 * kMU * hinv;
	// tau_b = 4.0 * kMU / 0.01 * detJb;

	/* Assemble the weak imposition of the boundary condition */
	if(elem_F) {
		elem_F += idx * NSHL * BS;
		for(I i = 0; i < NSHL * BS; ++i) {
			elem_F[i] = 0.0;
		}
		T tmp0[3];
		T tmp1[3 * 3];
		for(I iq = 0; iq < NQRB; ++iq) {
			uadv[0] = qr_wgalpha[NQRB * 0 + iq];
			uadv[1] = qr_wgalpha[NQRB * 1 + iq];
			uadv[2] = qr_wgalpha[NQRB * 2 + iq];
			T unor = uadv[0] * nv[0] + uadv[1] * nv[1] + uadv[2] * nv[2];
			T uneg = (unor - fabs(unor)) * 0.5;
			for(I i = 0; i < 3; ++i) {
				tmp0[i] = 0.0;
				tmp0[i] += nv[i] * qr_wgalpha[NQRB * 3 + iq];
				tmp0[i] -= kMU * (nv[0] * qr_wggradalpha[3 * i + 0] +
													nv[1] * qr_wggradalpha[3 * i + 1] +
													nv[2] * qr_wggradalpha[3 * i + 2]);
				tmp0[i] -= kMU * (nv[0] * qr_wggradalpha[3 * 0 + i] +
													nv[1] * qr_wggradalpha[3 * 1 + i] +
													nv[2] * qr_wggradalpha[3 * 2 + i]);
				tmp0[i] -= kRHO * uneg * uadv[i]; // qr_wgalpha[NQRB * i + iq];
				tmp0[i] += tau_b * uadv[i]; // qr_wgalpha[NQRB * i + iq];

			}	

			for(I i = 0; i < 3; ++i) {
				for(I j = 0; j < 3; ++j) {
					// tmp1[i * 3 + j] = -kMU * (nv[i] * qr_wgalpha[NQRB * j + iq] + nv[j] * qr_wgalpha[NQRB * i + iq]);
					tmp1[i * 3 + j] = -kMU * (nv[i] * uadv[j] + nv[j] * uadv[i]);
				}
			}

			for(I aa = 0; aa < NSHL; ++aa) {
				for(I ii = 0; ii < 3; ++ii) {
					T bm = 0.0;
					bm += c_shlub[NQRB * NSHL * iorn + iq * NSHL + aa] * tmp0[ii];
					bm += shgradg[aa * 3 + 0] * tmp1[ii * 3 + 0];
					bm += shgradg[aa * 3 + 1] * tmp1[ii * 3 + 1];
					bm += shgradg[aa * 3 + 2] * tmp1[ii * 3 + 2];
					elem_F[M2D(aa, ii)] += bm * c_gwb[iq];
				}
				elem_F[M2D(aa, 3)] -= c_shlub[NQRB * NSHL * iorn + iq * NSHL + aa] * unor * c_gwb[iq];
				// elem_F[M2D(aa, 3)] = 0.0;
				// for(I ii = 0; ii < 4; ++ii) {
				// 	if(idx == 13250 && isnan(elem_F[M2D(aa, ii)])) {
				// 		printf("idx = %d, iq = %d\n", idx, iq);
				// 		printf("elem_F[%d, %d] = %g\n", aa, ii, elem_F[M2D(aa, ii)]);
				// 		printf("tmp0 = %g, %g, %g\n", tmp0[0], tmp0[1], tmp0[2]);
				// 		printf("tmp1 = %g, %g, %g\n", tmp1[0], tmp1[1], tmp1[2]);
				// 		printf("tmp1 = %g, %g, %g\n", tmp1[3], tmp1[4], tmp1[5]);
				// 		printf("tmp1 = %g, %g, %g\n", tmp1[6], tmp1[7], tmp1[8]);
				// 	}
				// }
			}

		}
	}
	if(elem_J) {
		T fact1 = kALPHAM;
		T fact2 = kDT * kALPHAF * kGAMMA;
		T tmp0, tmp1;
		elem_J += idx * NSHL * NSHL * BS * BS;
		for(I i = 0; i < NSHL * NSHL * BS * BS; ++i) {
			elem_J[i] = 0.0;
		}
		T shnorm[NSHL];
		for(I aa = 0; aa < NSHL; ++aa) {
			shnorm[aa] = 0.0;
			shnorm[aa] += shgradg[aa * 3 + 0] * nv[0];
			shnorm[aa] += shgradg[aa * 3 + 1] * nv[1];
			shnorm[aa] += shgradg[aa * 3 + 2] * nv[2];
		}

		for(I iq = 0; iq < NQRB; ++iq) {
			uadv[0] = qr_wgalpha[NQRB * 0 + iq];
			uadv[1] = qr_wgalpha[NQRB * 1 + iq];
			uadv[2] = qr_wgalpha[NQRB * 2 + iq];
			T unor = uadv[0] * nv[0] + uadv[1] * nv[1] + uadv[2] * nv[2];
			T uneg = (unor - fabs(unor)) * 0.5;
			for(I aa = 0; aa < NSHL; ++aa) {
				for(I bb = 0; bb < NSHL; ++bb) {
					/* dRM/dU */
					tmp0 = 0.0;
					// tmp0 -= kMU * (nv[0] * shgradg[bb * 3 + 0] 
					// 						+ nv[1] * shgradg[bb * 3 + 1] 
					// 						+ nv[2] * shgradg[bb * 3 + 2]) * c_shlub[NQRB * NSHL * iorn + iq * NSHL + aa];
					// tmp0 -= kMU * (nv[0] * shgradg[aa * 3 + 0] 
					// 						+ nv[1] * shgradg[aa * 3 + 1] 
					// 						+ nv[2] * shgradg[aa * 3 + 2]) * c_shlub[NQRB * NSHL * iorn + iq * NSHL + bb];
					tmp0 -= kMU * (shnorm[bb] * c_shlub[NQRB * NSHL * iorn + iq * NSHL + aa] 
											+ shnorm[aa] * c_shlub[NQRB * NSHL * iorn + iq * NSHL + bb]);
					tmp0 -= kRHO * c_shlub[NQRB * NSHL * iorn + iq * NSHL + aa] 
											 * c_shlub[NQRB * NSHL * iorn + iq * NSHL + bb] * uneg;
					tmp0 += tau_b * c_shlub[NQRB * NSHL * iorn + iq * NSHL + aa] * c_shlub[NQRB * NSHL * iorn + iq * NSHL + bb];
					elem_J[M4D(aa, bb, 0, 0)] += fact2 * tmp0 * c_gwb[iq];
					elem_J[M4D(aa, bb, 1, 1)] += fact2 * tmp0 * c_gwb[iq];
					elem_J[M4D(aa, bb, 2, 2)] += fact2 * tmp0 * c_gwb[iq];
					// if(idx == 0 && aa == 0 && bb == 0) {
					// 		printf("sys = %.17e tauB= %.17e, uneg=%.17e, unor=%.17e\n", tmp0, tau_b, uneg, unor);
					// }
					
					for(I ii = 0; ii < 3; ++ii) {
						for(I jj = 0; jj < 3; ++jj) {
							tmp0 = 0.0;
							tmp0 -= kMU * c_shlub[NQRB * NSHL * iorn + iq * NSHL + aa] * shgradg[bb * 3 + ii] * nv[jj];
							tmp0 -= kMU * c_shlub[NQRB * NSHL * iorn + iq * NSHL + bb] * shgradg[aa * 3 + jj] * nv[ii];
							// if(idx == 0 && aa == 0 && bb == 0) {
							// 	printf("sys[%d][%d] = %.17e %.17e %.17e\n", ii, jj, fact2, tmp0, c_gwb[iq]);
							// }
							elem_J[M4D(aa, bb, ii, jj)] += fact2 * tmp0 * c_gwb[iq];
						}
					}

					tmp0 = c_shlub[NQRB * NSHL * iorn + iq * NSHL + aa] * c_shlub[NQRB * NSHL * iorn + iq * NSHL + bb];
					for(I ii = 0; ii < 3; ++ii) {
						/* dRC/dU */
						elem_J[M4D(aa, bb, 3, ii)] -= fact2 * tmp0 * nv[ii] * c_gwb[iq]; 
						/* dRM/dP */
						elem_J[M4D(aa, bb, ii, 3)] += tmp0 * nv[ii] * c_gwb[iq];
					}

				}
			}
		}
#ifdef DBG_TET
		if(idx == 0 && elem_J) {
			I aa = 0;
			I bb = 0;
			printf("num_thread = %d\n", blockDim.x * gridDim.x);
			printf("batch_size = %d\n", batch_size);
			printf("uadv=[%e %e %e]\n", uadv[0], uadv[1], uadv[2]);
			printf("nv=[%e %e %e]\n", nv[0], nv[1], nv[2]);
			printf("elem_J[M4D(%d, %d, :, :)] = \n", aa, bb);
			for(I ii = 0; ii < 4; ++ii) {
				for(I jj = 0; jj < 4; ++jj) {
					printf("%.17e ", elem_J[M4D(aa, bb, ii, jj)]);
				}
				printf("\n");
			}

		}
#endif

	}
}

template <typename I, typename T>
__global__ void GetElemInvJ3DLoadMatrixBatchKernel(I batch_size, T** A, T* elem_metric, I inc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batch_size) return;
	A[i] = elem_metric + i * inc;
}

template <typename I, typename T>
__global__ void GetStridedBatchKernel(I batch_size, T* begin, T** batch, I stride) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batch_size) return;
	batch[i] = begin + i * stride;
}

template <typename I, typename T>
__global__ void GetOffsetedBatchKernel(I batch_size, T* begin, T** batch, const I* offset, I stride) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batch_size) return;
	batch[i] = begin + offset[i] * stride;
}

template <typename I, typename T, typename INC>
__global__ void IncrementByN(I n, T* a, INC inc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n) return;
	a[i] += inc;
}


template <typename I, typename T, I NSHL=4>
void GetElemInvJ3D(I batch_size, const I* ien, const I* batch_index_ptr, const T* xg, T* elem_metric, void* buff) {
	f64 one = 1.0, zero = 0.0;
	int block_size = 256;
	int num_block = CEIL_DIV(batch_size, block_size);

	cudaStream_t stream[1] = {0};
	cublasHandle_t handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE);
	// cudaStreamCreate(stream + 0);


	/* 0. Load the elements in the batch into elem_metric[0:10, :]  */
	GetElemJ3DKernel<I, T, NSHL><<<num_block, block_size, 0, stream[0]>>>(batch_size, ien, batch_index_ptr, xg, elem_metric);

	byte* buff_ptr = (byte*)buff;

	T* d_elem_buff = (T*)buff_ptr;
	T** d_mat_batch = (T**)(buff_ptr + batch_size * 10 * SIZE_OF(T));
	T** d_matinv_batch = (T**)(buff_ptr + batch_size * 10 * SIZE_OF(T) + batch_size * SIZE_OF(T*));

	int* info = (int*)((byte*)d_matinv_batch + batch_size * SIZE_OF(T*));
	int* pivot = info + batch_size;

	/* Set mat_batch */
	// GetElemInvJ3DLoadMatrixBatchKernel<I, T><<<num_block, block_size, 0, stream[0]>>>(batch_size, d_matinv_batch, d_elem_buff, 10);
	GetStridedBatchKernel<I, T><<<num_block, block_size, 0, stream[0]>>>(batch_size, d_elem_buff, d_matinv_batch, 10);
	// GetElemInvJ3DLoadMatrixBatchKernel<I, T><<<num_block, block_size, 0, stream[0]>>>(batch_size, d_mat_batch, elem_metric, 10);	
	GetStridedBatchKernel<I, T><<<num_block, block_size, 0, stream[0]>>>(batch_size, elem_metric, d_mat_batch, 10);

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

template <typename I, typename M, typename T>
__global__ void SetupMaskKernel(I batch_size, const I* batch_index_ptr, const T* value, T target, M* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batch_size) return;
	T v = value[batch_index_ptr[i]];
	mask[i] = M{v == target};
}

template <typename I, typename T>
__global__ void SetupShlubBatchKernel(I batch_size, const I* batch_index_ptr, T* begin, T** elem_shlu, int stride) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batch_size) return;
	elem_shlu[i] = begin + batch_index_ptr[i] * stride;
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
										 const T* elem_G, const T* shgradg,
										 const T* qr_wgalpha, const T* qr_dwgalpha, const T* qr_wggradalpha,
										 T* elem_F, T* elem_J, cudaStream_t stream) {

	const int NSHL = 4;
	if(elem_F) {
		i32 block_dim = 256;
		i32 grid_dim = CEIL_DIV(batch_size, block_dim);
		AssembleWeakFormKernel<I, T, 4, 1><<<grid_dim, block_dim>>>(
				batch_size,
				elem_G, shgradg,
				qr_wgalpha, qr_dwgalpha, qr_wggradalpha,
				elem_F, NULL);
		if(0) {
			f64* h_elem_F = (f64*)malloc(batch_size * BS * NSHL * SIZE_OF(f64));
			cudaMemcpy(h_elem_F, elem_F, batch_size * BS * NSHL * SIZE_OF(f64), cudaMemcpyDeviceToHost);
			FILE* fp = fopen("elem_F_shared.txt", "w");
			for(index_type i = 0; i < batch_size * NSHL; ++i) {
				for(index_type j = 0; j < BS; ++j) {
					fprintf(fp, "%.17e ", h_elem_F[i * BS + j]);
				}
				fprintf(fp, "\n");
			
			}	
			fclose(fp);
			free(h_elem_F);
		}
	}
	if(elem_J) {
		if(1) {
			const i32 block_dim = 64*4;
			i32 grid_dim = CEIL_DIV(batch_size * NSHL * NSHL, block_dim);

			cudaMemsetAsync(elem_J, 0, batch_size * NSHL * NSHL * BS * BS * SIZE_OF(T), stream);
			AssembleWeakFormLHSKernel<I, T, 4, block_dim><<<grid_dim, block_dim, 0, stream>>>(
					batch_size,
					elem_G, shgradg,
					qr_wgalpha, qr_dwgalpha, qr_wggradalpha,
					elem_J);
			if(0) {
				f64* h_elem_J = (f64*)malloc(batch_size * BS * NSHL * BS * NSHL * SIZE_OF(f64));
				cudaMemcpy(h_elem_J, elem_J, batch_size * BS * NSHL * BS * NSHL * SIZE_OF(f64), cudaMemcpyDeviceToHost);
				f64 nrm = 0.0;
				for(index_type i = 0; i < batch_size * BS * NSHL * BS * NSHL; i += BS * BS) {
					// nrm += h_elem_J[i] * h_elem_J[i];
					nrm += fabs(h_elem_J[i + 3]);
					nrm += fabs(h_elem_J[i + 9]);
					nrm += fabs(h_elem_J[i + 15]);
				}
				printf("NRM: %a %.17e\n", nrm, nrm);
				FILE* fp = fopen("elem_J_shared.txt", "w");
				for(index_type i = 0; i < batch_size * NSHL * NSHL; ++i) {
					for(index_type j = 0; j < BS * BS; ++j) {
						fprintf(fp, "%.17e ", h_elem_J[i * BS * BS + j]);
					}
					fprintf(fp, "\n");
				}
				fclose(fp);
				free(h_elem_J);
				exit(-1);
			}

		}
		if(0){
			i32 block_dim = 256;
			i32 grid_dim = CEIL_DIV(batch_size, block_dim);

			cudaMemsetAsync(elem_J, 0, batch_size * NSHL * NSHL * BS * BS * SIZE_OF(T), stream);
			AssembleWeakFormKernel<I, T, 4, 2><<<grid_dim, block_dim, 0, stream>>>(
					batch_size,
					elem_G, shgradg,
					qr_wgalpha, qr_dwgalpha, qr_wggradalpha,
					NULL, elem_J);
			if(0) {
				f64* h_elem_J = (f64*)malloc(batch_size * BS * NSHL * BS * NSHL * SIZE_OF(f64));
				cudaMemcpy(h_elem_J, elem_J, batch_size * BS * NSHL * BS * NSHL * SIZE_OF(f64), cudaMemcpyDeviceToHost);
				FILE* fp = fopen("elem_J_shared.txt", "w");
				for(index_type i = 0; i < NSHL * NSHL; ++i) {
					for(index_type j = 0; j < BS * BS; ++j) {
						fprintf(fp, "%.17e ", h_elem_J[i * BS * BS + j]);
					}
					fprintf(fp, "\n");
				}
				fclose(fp);
				free(h_elem_J);
			}

		}
		// cudaStreamSynchronize(stream);
		// exit(0);
	}
}

template <typename I, typename T>
void FaceAssembly(I batch_size, const T* elem_invJ, const T* nv, const T* shgradg,
									const I* forn, const T* qr_wgalpha, const T* qr_wggradalpha,
									T* elem_F, T* elem_J) {
	i32 block_dim = 256;
	i32 grid_dim = CEIL_DIV(batch_size, block_dim);
	const int NSHL = 4;
	FaceAssemblyKernel<I, T><<<grid_dim, block_dim>>>(batch_size, elem_invJ, nv, shgradg, forn, qr_wgalpha, qr_wggradalpha, elem_F, elem_J);
#ifdef DBG_TET
	if(elem_J) {
		f64* h_elem_J = (f64*)malloc(batch_size * BS * NSHL * BS * NSHL * SIZE_OF(f64));
		cudaMemcpy(h_elem_J, elem_J, batch_size * BS * NSHL * BS * NSHL * SIZE_OF(f64), cudaMemcpyDeviceToHost);
		f64 nrm = 0.0;
		for(index_type i = 0; i < batch_size * BS * NSHL * BS * NSHL; i += BS * BS) {
			// nrm += h_elem_J[i] * h_elem_J[i];
			nrm += fabs(h_elem_J[i + 3]);
			nrm += fabs(h_elem_J[i + 9]);
			nrm += fabs(h_elem_J[i + 15]);
		}
		printf("batch_size = %d\n", batch_size);
		printf("NRM: %a %.17e\n", nrm, nrm);
		FILE* fp = fopen("elem_SJ_shared.txt", "w");
		for(index_type i = 0; i < batch_size * NSHL * NSHL; ++i) {
			for(index_type j = 0; j < BS * BS; ++j) {
				fprintf(fp, "%.17e ", h_elem_J[i * BS * BS + j]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
		free(h_elem_J);
		// exit(-1);
	}
#endif
}

__BEGIN_DECLS__





void AssembleSystemTet(Mesh3D *mesh,
											 /* Field *wgold, Field* dwgold, Field* dwg, */
											 f64* wgalpha_dptr, f64* dwgalpha_dptr,
											 f64* F, Matrix* J) {

	const i32 NSHL = 4;
	const i32 MAX_BATCH_SIZE = 1 << 10;
	index_type num_tet = Mesh3DNumTet(mesh);
	index_type num_node = Mesh3DNumNode(mesh);
	const Mesh3DData* dev = Mesh3DDevice(mesh);
	const index_type* ien = Mesh3DDataTet(dev);
	const f64* xg = Mesh3DDataCoord(dev);
	cublasHandle_t handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE);

	UNUSED(MAX_BATCH_SIZE);
	UNUSED(num_tet);
	UNUSED(num_node);

	// f64* wgold_dptr = ArrayData(FieldDevice(wgold));
	// f64* dwgold_dptr = ArrayData(FieldDevice(dwgold));
	// f64* dwg_dptr = ArrayData(FieldDevice(dwg));



	f64* d_shlgradu = NULL, *d_shlu = NULL, *d_gw = NULL;
	if(!d_shlgradu) {
		d_shlgradu = (f64*)CdamMallocDevice(NSHL * 3 * SIZE_OF(f64));
		cudaMemcpy(d_shlgradu, h_shlgradu, NSHL * 3 * SIZE_OF(f64), cudaMemcpyHostToDevice);
	}
	if(!d_shlu) {
		d_shlu = (f64*)CdamMallocDevice(NQR * NSHL * SIZE_OF(f64));
		cudaMemcpy(d_shlu, h_shlu, NQR * NSHL * SIZE_OF(f64), cudaMemcpyHostToDevice);
	}
	if(!d_gw) {
		d_gw = (f64*)CdamMallocDevice(NQR * SIZE_OF(f64));
		cudaMemcpy(d_gw, h_gw, NQR * SIZE_OF(f64), cudaMemcpyHostToDevice);
	}
	ASSERT(F || J && "Either F or J should be provided");
	printf("Assemble: %s %s\n", F ? "F" : "", J ? "J" : "");


	index_type* batch_index_ptr = NULL;
	index_type max_batch_size = 0, batch_size = 0;
	f64 one = 1.0, zero = 0.0, minus_one = -1.0;


	f64 fact1 = kDT * kALPHAF * kGAMMA, fact2 = kDT * kALPHAF * (1.0 - kGAMMA); 
	// f64* wgalpha_dptr = (f64*)CdamMallocDevice(num_node * SIZE_OF(f64));
	/* wgalpha_ptr[:] = wgold[:] + kDT * kALPHAF * (1.0 - kGAMMA) * dwgold[:] + kDT * kALPHAF * kGAMMA * dwg[:] */
	// cublasDcopy(handle, num_node, wgold_dptr, 1, wgalpha_dptr, 1);
	// cublasDaxpy(handle, num_node, &fact1, dwg_dptr, 1, wgalpha_dptr, 1);
	// cublasDaxpy(handle, num_node, &fact2, dwgold_dptr, 1, wgalpha_dptr, 1);


	for(index_type b = 0; b < mesh->num_batch; ++b) {
		// batch_size = CountValueColorLegacy(mesh->color, num_tet, b);
		// ASSERT(batch_size == mesh->batch_offset[b+1] - mesh->batch_offset[b]);
		batch_size = mesh->batch_offset[b+1] - mesh->batch_offset[b];
		if (batch_size > max_batch_size) {
			max_batch_size = batch_size;
		}
	}
	// max_batch_size = CEIL_DIV(max_batch_size, 16) * 16;

	// batch_index_ptr = (index_type*)CdamMallocDevice(max_batch_size * SIZE_OF(index_type));

	f64* buffer = (f64*)CdamMallocDevice(max_batch_size * SIZE_OF(f64) * NSHL * BS);
	f64* elem_invJ = (f64*)CdamMallocDevice(max_batch_size * SIZE_OF(f64) * (3 * 3 + 1));
	f64* shgradg = (f64*)CdamMallocDevice(max_batch_size * (SIZE_OF(f64) * NSHL * 3 + 4 * SIZE_OF(int)));

	f64* qr_wgalpha = (f64*)CdamMallocDevice(max_batch_size * SIZE_OF(f64) * NQR * BS);
	f64* qr_dwgalpha = (f64*)CdamMallocDevice(max_batch_size * SIZE_OF(f64) * NQR * BS);
	f64* qr_wggradalpha = (f64*)CdamMallocDevice(max_batch_size * SIZE_OF(f64) * 3 * BS);

	i32 num_thread = 256;
	// i32 num_block = (max_batch_size + num_thread - 1) / num_thread;

	// f64* elem_buff = (f64*)CdamMallocDevice(max_batch_size * SIZE_OF(f64) * NSHL * NSHL);
	f64* elem_F = NULL;
	f64* elem_J = NULL;

	if(F) {
		elem_F = (f64*)CdamMallocDevice(max_batch_size * SIZE_OF(f64) * NSHL * BS);
	}
	if(J) {
		elem_J = (f64*)CdamMallocDevice(max_batch_size * SIZE_OF(f64) * NSHL * NSHL * BS * BS);
	}

	f64 time_len[6] = {0, 0, 0, 0, 0, 0};
	std::chrono::steady_clock::time_point start, end;

	/* Assume all 2D arraies are column-majored */
	for(index_type b = 0; b < mesh->num_batch; ++b) {

		// batch_size = CountValueColorLegacy(mesh->color, num_tet, b);
		batch_size = mesh->batch_offset[b+1] - mesh->batch_offset[b];
		// batch_size = CountValueColor(mesh->color, num_tet, b, count_value_buffer);

		if(batch_size == 0) {
			break;
		}


		// FindValueColor(mesh->color, num_tet, b, batch_index_ptr);
		// cudaMemcpy(batch_index_ptr, mesh->batch_ind + mesh->batch_offset[b], batch_size * SIZE_OF(index_type), cudaMemcpyDeviceToDevice);
		batch_index_ptr = mesh->batch_ind + mesh->batch_offset[b];
		// CUGUARD(cudaGetLastError());
		/* 0. Calculate the element metrics */
		/* 0.0. Get dxi/dx and det(dxi/dx) */

		start = std::chrono::steady_clock::now();
		GetElemInvJ3D<index_type, f64, NSHL>(batch_size, ien, batch_index_ptr, xg, elem_invJ, shgradg);
		end = std::chrono::steady_clock::now();
		time_len[0] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		/* 0.1. Calculate the gradient of the shape functions */
		start = std::chrono::steady_clock::now();
		GetShapeGradKernel<index_type, f64><<<CEIL_DIV(batch_size, num_thread), num_thread>>>(batch_size, elem_invJ, shgradg);
		/* 0.2. Calculate the metric of the elements */
		/* elem_invJ[0:3, 0:3, e] = shgradg[0:3, 1:NSHL, e].T @ shgradg[0:3, 1:NSHL, e] for e in range(batch_size) */
		cublasDgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
															3, 3, 3,
															&one,
															shgradg + 3, 3, (long long)(3 * NSHL),
															shgradg + 3, 3, (long long)(3 * NSHL),
															&zero,
															elem_invJ, 3, (long long)(3 * 3 + 1),
															batch_size);
		end = std::chrono::steady_clock::now();
		time_len[1] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	
		start = std::chrono::steady_clock::now();
	
		/* 1.0. Load the field value on the vertices */
		/* Load velocity */
		LoadElementValueKernel<index_type, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 3,
																																																	 batch_index_ptr, ien,
																																																	 wgalpha_dptr, 3,
																																																	 buffer, BS * NSHL);
		/* Load pressure */
		LoadElementValueKernel<index_type, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 1,
																																																		batch_index_ptr, ien,
																																																		dwgalpha_dptr + num_node * 3, 1,
																																																		buffer + 3 * NSHL, BS * NSHL);
		/* Load phi */
		LoadElementValueKernel<index_type, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 1,
																																																	 batch_index_ptr, ien,
																																																	 wgalpha_dptr + num_node * 4, 1,
																																																	 buffer + 4 * NSHL, BS * NSHL);
		/* Load temperature */
		LoadElementValueKernel<index_type, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 1,
																																																	 batch_index_ptr, ien,
																																																	 wgalpha_dptr + num_node * 5, 1,
																																																	 buffer + 5 * NSHL, BS * NSHL);
		// thrust::for_each(thrust::device,
		// 								 thrust::make_counting_iterator<index_type>(0),
		// 								 thrust::make_counting_iterator<index_type>(batch_size),
		// 								 LoadBatchFunctor<index_type, f64, NSHL, 1>(batch_index_ptr, ien, wgalpha_dptr, buffer));
											
		// CUGUARD(cudaGetLastError());
		/* 1.1. Calculate the gradient*/
		/* qr_wggradalpha[0:3, 0:BS, 0:batch_size] = sum(shgradg[0:3, 0:NSHL, 0:batch_size] * buffer[0:NSHL, 0:BS, 0:batch_size], axis=1) */
		cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
															3, BS, NSHL,
															&one,
															shgradg, 3, (long long)(NSHL * 3),
															buffer, NSHL, (long long)(NSHL * BS),
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
		// cudaMemset(qr_dwgalpha, 0, max_batch_size * SIZE_OF(f64));
		LoadElementValueKernel<index_type, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 3,
																																																	 batch_index_ptr, ien,
																																																	 dwgalpha_dptr, 3,
																																																	 buffer, BS * NSHL);
		LoadElementValueKernel<index_type, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 1,
																																																	 batch_index_ptr, ien,
																																																	 dwgalpha_dptr + num_node * 3, 1,
																																																	 buffer + 3 * NSHL, BS * NSHL);
		LoadElementValueKernel<index_type, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 1,
																																																	 batch_index_ptr, ien,
																																																	 dwgalpha_dptr + num_node * 4, 1,
																																																	 buffer + 4 * NSHL, BS * NSHL);
		LoadElementValueKernel<index_type, f64, NSHL><<<CEIL_DIV(batch_size * NSHL, num_thread), num_thread>>>(batch_size, 1,
																																																	 batch_index_ptr, ien,
																																																	 dwgalpha_dptr + num_node * 5, 1,
																																																	 buffer + 5 * NSHL, BS * NSHL);
		// thrust::for_each(thrust::device,
		// 								 thrust::make_counting_iterator<index_type>(0),
		// 								 thrust::make_counting_iterator<index_type>(batch_size),
		// 								 LoadBatchFunctor<index_type, f64, NSHL, 1>(batch_index_ptr, ien, dwgalpha_dptr, buffer));
		// CUGUARD(cudaGetLastError());
		/* qr_dwgalpha[0:NQR, 0:BS, 0:batch_size] = d_shlu[0:NSHL, 0:NQR].T @ buffer[0:NSHL, 0:BS, 0:batch_size] */
		cublasDgemm(handle,
								CUBLAS_OP_T, CUBLAS_OP_N,
								NQR, batch_size * BS, NSHL,
								&one,
								d_shlu, NSHL,
								buffer, NSHL,
								&zero,
								qr_dwgalpha,
								NQR);
		end = std::chrono::steady_clock::now();
		// CUGUARD(cudaGetLastError());
		time_len[3] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		/* 2. Calculate the elementwise residual and jacobian */
		start = std::chrono::steady_clock::now();
		IntElemAssembly<index_type, f64>(batch_size,
															elem_invJ, shgradg,
															qr_wgalpha, qr_dwgalpha, qr_wggradalpha,
															elem_F, elem_J, 0);
		end = std::chrono::steady_clock::now();
		time_len[4] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		/* 3. Assemble the global residual vector */
		start = std::chrono::steady_clock::now();
		if(F) {
			ElemRHSLocal2Global(batch_size, batch_index_ptr, ien,
												 3, 
												 elem_F, BS,
												 F, 3, (const index_type*)NULL);
			ElemRHSLocal2Global(batch_size, batch_index_ptr, ien,
												  1, 
												  elem_F + 3, BS,
												  F + num_node * 3, 1, (const index_type*)NULL);
			ElemRHSLocal2Global(batch_size, batch_index_ptr, ien,
													1, 
													elem_F + 4, BS,
													F + num_node * 4, 1, (const index_type*)NULL);
			ElemRHSLocal2Global(batch_size, batch_index_ptr, ien,
													1, 
													elem_F + 5, BS,
													F + num_node * 5, 1, (const index_type*)NULL);
		}
		/* 4. Assemble the global residual matrix */
		if(J) {
			// MatrixAddElementLHS(J, NSHL, BS, batch_size, ien, batch_index_ptr, elem_J, BS * NSHL);
			// thrust::fill(thrust::device, elem_J, elem_J + batch_size * BS * NSHL * BS * NSHL, 1.0);
			ElemLHSLocal2GlobalBlocked(batch_size, batch_index_ptr, ien,
																 BS, BS,
																 elem_J, J, (const index_type*)NULL);

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

	CdamFreeDevice(d_shlgradu, NSHL * 3 * SIZE_OF(f64));
	CdamFreeDevice(d_shlu, NQR * NSHL * SIZE_OF(f64));
	CdamFreeDevice(d_gw, NQR * SIZE_OF(f64));

	CdamFreeDevice(buffer, max_batch_size * SIZE_OF(f64) * NSHL * BS);
	CdamFreeDevice(elem_invJ, max_batch_size * SIZE_OF(f64) * 10);
	CdamFreeDevice(shgradg, max_batch_size * (SIZE_OF(f64) * NSHL * 3 + 4 * SIZE_OF(int)));


	CdamFreeDevice(elem_F, max_batch_size * SIZE_OF(f64) * NSHL * BS);
	CdamFreeDevice(elem_J, max_batch_size * SIZE_OF(f64) * NSHL * NSHL * BS * BS);

	CdamFreeDevice(qr_wgalpha, max_batch_size * SIZE_OF(f64) * NQR * BS);
	CdamFreeDevice(qr_dwgalpha, max_batch_size * SIZE_OF(f64) * NQR * BS);
	CdamFreeDevice(qr_wggradalpha, max_batch_size * SIZE_OF(f64) * 3 * BS);
	// CdamFreeDevice(batch_index_ptr, max_batch_size * SIZE_OF(index_type));
}

void AssembleSystemTetFace(Mesh3D* mesh,
													 f64* wgalpha_dptr, f64* dwgalpha_dptr,
													 f64* F, Matrix* J) {
	const i32 NSHL = 4;
	const i32 MAX_BATCH_SIZE = 1 << 10;
	const Mesh3DData* dev = Mesh3DDevice(mesh);
	const index_type num_tet = Mesh3DNumTet(mesh);
	const index_type num_node = Mesh3DNumNode(mesh);
	const index_type* ien = Mesh3DDataTet(dev);
	const f64* xg = Mesh3DDataCoord(dev);
	const color_t* color = mesh->color;
	color_t num_color = mesh->num_color;
	color_t c;
	f64 one = 1.0, zero = 0.0;

	cublasHandle_t handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE);

	// static f64* d_shlgradu = NULL, *d_shlu = NULL;
	// if(!d_shlgradu) {
	// 	d_shlgradu = (f64*)CdamMallocDevice(NSHL * 3 * SIZE_OF(f64));
	// 	cudaMemcpy(d_shlgradu, h_shlgradu, NSHL * 3 * SIZE_OF(f64), cudaMemcpyHostToDevice);
	// }
	// if(!d_shlu) {
	// 	d_shlu = (f64*)CdamMallocDevice(NQR * NSHL * SIZE_OF(f64));
	// 	cudaMemcpy(d_shlu, h_shlu, NQR * NSHL * SIZE_OF(f64), cudaMemcpyHostToDevice);
	// }
	f64* d_shlub = (f64*)CdamMallocDevice(NQRB * NSHL * NSHL * SIZE_OF(f64));
	cudaMemcpy(d_shlub, h_shlub, NQRB * NSHL * NSHL * SIZE_OF(f64), cudaMemcpyHostToDevice);

	index_type b;
	index_type max_num_facet = 0;
	value_type* buffer, *elem_invJ, *nv, *shgradg, *qr_wgalpha, *qr_wggradalpha;
	value_type* elem_F = NULL, *elem_J = NULL;
	index_type* mask = NULL;
	for(b = 0; b < mesh->num_bound; ++b) {
		if(Mesh3DBoundNumElem(mesh, b) > max_num_facet) {
			max_num_facet = Mesh3DBoundNumElem(mesh, b);
		}
	}

	buffer = (value_type*)CdamMallocDevice(max_num_facet * SIZE_OF(value_type) * NSHL * BS);
	elem_invJ = (value_type*)CdamMallocDevice(max_num_facet * SIZE_OF(value_type) * (3 * 3 + 1));
	nv = (value_type*)CdamMallocDevice(max_num_facet * SIZE_OF(value_type) * 3);
	shgradg = (value_type*)CdamMallocDevice(max_num_facet * (SIZE_OF(value_type) * NSHL * 3 + 4 * SIZE_OF(int)));
	
	qr_wgalpha = (value_type*)CdamMallocDevice(max_num_facet * SIZE_OF(value_type) * NQRB * BS);
	qr_wggradalpha = (value_type*)CdamMallocDevice(max_num_facet * SIZE_OF(value_type) * 3 * BS);

	mask = (index_type*)CdamMallocDevice(max_num_facet * SIZE_OF(index_type));

	f64** d_shlub_batch = (f64**)CdamMallocDevice(max_num_facet * SIZE_OF(f64*));
	f64** d_buffer_batch = (f64**)CdamMallocDevice(max_num_facet * SIZE_OF(f64*));
	f64** d_qr_wgalpha_batch = (f64**)CdamMallocDevice(max_num_facet * SIZE_OF(f64*));

	if(F) {
		elem_F = (value_type*)CdamMallocDevice(max_num_facet * SIZE_OF(value_type) * NSHL * BS);
	}
	if(J) {
		elem_J = (value_type*)CdamMallocDevice(max_num_facet * SIZE_OF(value_type) * NSHL * NSHL * BS * BS);
	}

	for(b = 0; b < mesh->num_bound; b++) {
#ifndef DBG_TET
		if(b != 4) continue;
#endif

		index_type num_face = Mesh3DBoundNumElem(mesh, b);
		const index_type* f2e = Mesh3DBoundF2E(mesh, b);
		const index_type* forn = Mesh3DBoundFORN(mesh, b);

		/* 0. Calculate the element metrics */
		// GetElemJ3DKernel<index_type, value_type, NSHL><<<CEIL_DIV(num_face, 256), 256>>>(num_face, ien, f2e, xg, elem_invJ);
		GetElemInvJ3D<index_type, value_type, NSHL>(num_face, ien, f2e, xg, elem_invJ, shgradg);
		GetShapeGradKernel<index_type, value_type><<<CEIL_DIV(num_face, 256), 256>>>(num_face, elem_invJ, shgradg);
		GetElemFaceNVKernel<index_type, value_type><<<CEIL_DIV(num_face, 256), 256>>>(num_face, elem_invJ, forn, nv);

		/* 1. Interpolate the field values */
		LoadElementValueKernel<index_type, value_type, NSHL><<<CEIL_DIV(num_face * NSHL, 256), 256>>>(num_face, 3,
																																																	f2e, ien,
																																																	wgalpha_dptr, 3,
																																																	buffer, BS * NSHL);
		LoadElementValueKernel<index_type, value_type, NSHL><<<CEIL_DIV(num_face * NSHL, 256), 256>>>(num_face, 1,
																																																	f2e, ien,
																																																	dwgalpha_dptr + num_node * 3, 1,
																																																	buffer + 3 * NSHL, BS * NSHL);
		/* 1.1. Calculate the gradient */
		cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
															3, BS, NSHL,
															&one,
															shgradg, 3, (long long)(NSHL * 3),
															buffer, NSHL, (long long)(NSHL * BS),
															&zero,
															qr_wggradalpha, 3, (long long)(BS * 3),
															num_face);

		/* 1.2. Calculate the field value on the quadrature */
		/* set up d_shlub_batch[:] = d_shlub + forn[:] * 12 */
		/* set up d_buffer_batch[:] = buffer + (0:num_face) * NSHL * BS */
		/* set up d_qr_wgalpha_batch[:] = qr_wgalpha + (0:num_face) * NQRB * BS */
		GetOffsetedBatchKernel<index_type, f64><<<CEIL_DIV(num_face, 256), 256>>>(num_face, d_shlub, d_shlub_batch, forn, NSHL * NQRB);
		GetStridedBatchKernel<index_type, f64><<<CEIL_DIV(num_face, 256), 256>>>(num_face, buffer, d_buffer_batch, NSHL * BS);
		GetStridedBatchKernel<index_type, f64><<<CEIL_DIV(num_face, 256), 256>>>(num_face, qr_wgalpha, d_qr_wgalpha_batch, NQRB * BS);
		/* qr_wgalpha[0:NQRB, 0:BS, e] = d_shlub_batch[0:NSHL, 0:NQRB, e].T @ buffer[0:NSHL, 0:BS, e] for e in range(num_face) */
		cublasDgemmBatched(handle,
											 CUBLAS_OP_T, CUBLAS_OP_N,
											 NQRB, BS, NSHL,
											 &one,
											 (const f64* const*)d_shlub_batch, (long long)NSHL,
											 (const f64* const*)d_buffer_batch, (long long)NSHL,
											 &zero,
											 (f64**)d_qr_wgalpha_batch, NQRB,
											 num_face);
		/*
		cublasDgemm(handle,
								CUBLAS_OP_T, CUBLAS_OP_N,
								NQRB, num_face * BS, NSHL,
								&one,
								d_shlub, NSHL,
								buffer, NSHL,
								&zero,
								qr_wgalpha, NQRB);
		*/
		/* 2. Calculate the elementwise residual and jacobian */
		FaceAssembly(num_face, elem_invJ, nv, shgradg, forn, qr_wgalpha, qr_wggradalpha, elem_F, elem_J);

		/* 3. Assemble the global residual vector */

		if(F) {
			if(0) {
				f64 sum = thrust::reduce(thrust::device, elem_F, elem_F + num_face * NSHL * BS, 0.0);
				printf("sum: %e\n", sum);
				f64* h_nv = (f64*)malloc(num_face * 3 * SIZE_OF(f64));
				cudaMemcpy(h_nv, nv, num_face * 3 * SIZE_OF(f64), cudaMemcpyDeviceToHost);
				FILE* fp = fopen("nv.dat", "w");
				for(i32 i = 0; i < num_face; ++i) {
					fprintf(fp, "%16.14e %16.14e %16.14e\n", h_nv[i * 3], h_nv[i * 3 + 1], h_nv[i * 3 + 2]);
				}
				fclose(fp);
				free(h_nv);

				f64* h_elem_F = (f64*)malloc(num_face * NSHL * BS * SIZE_OF(f64));
				cudaMemcpy(h_elem_F, elem_F, num_face * NSHL * BS * SIZE_OF(f64), cudaMemcpyDeviceToHost);
				fp = fopen("elem_F.dat", "w");
				for(i32 i = 0; i < num_face; ++i) {
					for(i32 j = 0; j < NSHL * BS; ++j) {
						fprintf(fp, "%16.14e ", h_elem_F[i * NSHL * BS + j]);
					}
					fprintf(fp, "\n");
				}
				fclose(fp);
				free(h_elem_F);
			}
			for(c = 0; c < num_color; c++) {
				SetupMaskKernel<index_type><<<CEIL_DIV(num_face, 256), 256>>>(num_face, f2e, color, c, mask);
				ElemRHSLocal2Global<index_type, value_type>(num_face, f2e, ien,
																										3, 
																										elem_F, BS,
																										F, 3, mask);
				ElemRHSLocal2Global<index_type, value_type>(num_face, f2e, ien,
																										1, 
																										elem_F + 3, BS,
																										F + num_node * 3, 1, mask);
				ElemRHSLocal2Global<index_type, value_type>(num_face, f2e, ien,
																										1, 
																										elem_F + 4, BS,
																										F + num_node * 4, 1, mask);
				ElemRHSLocal2Global<index_type, value_type>(num_face, f2e, ien,
																										1, 
																										elem_F + 5, BS,
																										F + num_node * 5, 1, mask);
			}
		}

		if(J) {
			for(c = 0; c < num_color; c++) {
				SetupMaskKernel<index_type><<<CEIL_DIV(num_face, 256), 256>>>(num_face, f2e, color, c, mask);
				ElemLHSLocal2GlobalBlocked<index_type, value_type>(num_face, f2e, ien,
																													 BS, BS,
																													 elem_J, J, mask);

			}
		}
		
	}
	CdamFreeDevice(d_shlub, NQRB * NSHL * NSHL * SIZE_OF(f64));
	CdamFreeDevice(buffer, max_num_facet * SIZE_OF(value_type) * NSHL * BS);
	CdamFreeDevice(elem_invJ, max_num_facet * SIZE_OF(value_type) * 10);
	CdamFreeDevice(nv, max_num_facet * SIZE_OF(value_type) * 3);
	CdamFreeDevice(shgradg, max_num_facet * (SIZE_OF(value_type) * NSHL * 3 + 4 * SIZE_OF(int)));
	CdamFreeDevice(qr_wgalpha, max_num_facet * SIZE_OF(value_type) * NQR * BS);
	CdamFreeDevice(qr_wggradalpha, max_num_facet * SIZE_OF(value_type) * 3 * BS);

	CdamFreeDevice(d_shlub_batch, max_num_facet * SIZE_OF(f64*));
	CdamFreeDevice(d_buffer_batch, max_num_facet * SIZE_OF(f64*));
	CdamFreeDevice(d_qr_wgalpha_batch, max_num_facet * SIZE_OF(f64*));

	CdamFreeDevice(mask, max_num_facet * SIZE_OF(index_type));

	CdamFreeDevice(elem_F, max_num_facet * SIZE_OF(value_type) * NSHL * BS);
	CdamFreeDevice(elem_J, max_num_facet * SIZE_OF(value_type) * NSHL * NSHL * BS * BS);
}
__END_DECLS__
