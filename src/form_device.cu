
#include "common.h"
#include "alloc.h"
#include "form.h"


#define kMU (10.0 / 3.0)
#define kRHO (1e3)
#define kCP (1e3)
#define kKAPPA (1e-2)

__constant__ value_type fb[] = {0.0, 0.0, 0.0}; 
__constant__ value_type gw[] = {0.0416666666666667, 0.0416666666666667, 0.0416666666666667, 0.0416666666666667};
__constant__ value_type shlu[] = {0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105,
																	0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105,
																	0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105,
																	0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685};

const value_type h_gw[4] = {0.0416666666666667, 0.0416666666666667, 0.0416666666666667, 0.0416666666666667};

const value_type h_shlu[16] = {0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105,
														 0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105,
														 0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105,
														 0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685};

const value_type h_shlgradu[12] = {-1.0, -1.0, -1.0,
																	 1.0, 0.0, 0.0,
																	 0.0, 1.0, 0.0,
																	 0.0, 0.0, 1.0};


__constant__ value_type gwb[] = {0.1666666666666667, 0.1666666666666667, 0.1666666666666667};
__constant__ value_type shlub[][12] = {
	{0.0, 0.1666666666666667, 0.1666666666666667, 0.6666666666666667,
	 0.0, 0.1666666666666667, 0.6666666666666667, 0.1666666666666667,
	 0.0, 0.6666666666666667, 0.1666666666666667, 0.1666666666666667},

	{0.1666666666666667, 0.0, 0.1666666666666667, 0.6666666666666667,
	 0.1666666666666667, 0.0, 0.6666666666666667, 0.1666666666666667,
	 0.6666666666666667, 0.0, 0.1666666666666667, 0.1666666666666667},

	{0.1666666666666667, 0.1666666666666667, 0.0, 0.6666666666666667,
	 0.1666666666666667, 0.6666666666666667, 0.0, 0.1666666666666667,
	 0.6666666666666667, 0.1666666666666667, 0.0, 0.1666666666666667},

	{0.1666666666666667, 0.1666666666666667, 0.6666666666666667, 0.0,
	 0.1666666666666667, 0.6666666666666667, 0.1666666666666667, 0.0,
	 0.6666666666666667, 0.1666666666666667, 0.1666666666666667, 0.0}
};

const value_type h_gwb[] = {0.1666666666666667, 0.1666666666666667, 0.1666666666666667};

const value_type h_shlub[][12] = {
	{0.0, 0.1666666666666667, 0.1666666666666667, 0.6666666666666667,
	 0.0, 0.1666666666666667, 0.6666666666666667, 0.1666666666666667,
	 0.0, 0.6666666666666667, 0.1666666666666667, 0.1666666666666667},

	{0.1666666666666667, 0.0, 0.1666666666666667, 0.6666666666666667,
	 0.1666666666666667, 0.0, 0.6666666666666667, 0.1666666666666667,
	 0.6666666666666667, 0.0, 0.1666666666666667, 0.1666666666666667},

	{0.1666666666666667, 0.1666666666666667, 0.0, 0.6666666666666667,
	 0.1666666666666667, 0.6666666666666667, 0.0, 0.1666666666666667,
	 0.6666666666666667, 0.1666666666666667, 0.0, 0.1666666666666667},

	{0.1666666666666667, 0.1666666666666667, 0.6666666666666667, 0.0,
	 0.1666666666666667, 0.6666666666666667, 0.1666666666666667, 0.0,
	 0.6666666666666667, 0.1666666666666667, 0.1666666666666667, 0.0}};

/* Facet normal vectors of tet in the reference coordinate*/
/* Reserved for the normal vectors in the physical coordinate using Nanson's formula */
__constant__ value_type c_nv2[4*3] = { /* 0.57735026919, 0.57735026919, 0.57735026919, */
																			1.0, 1.0, 1.0,
																		 -1.0, 0.0, 0.0,
																			0.0, -1.0, 0.0,
																			0.0, 0.0, -1.0};

#define NQR(x) (sizeof(x) / sizeof(x[0]))

#define M2D(i, j) ((i) * BS + (j))
#define M4D(i, j, k, l) ((i) * BS * BS * BS + (j) * BS * BS + (k) * BS + (l))

__BEGIN_DECLS__

#ifdef CDAM_USE_CUDA
/** Load the value from the source array to the destination array belonging to the batch
 * The source array is columned-majored of size (max(elem_size, stridex), num_node)
 * The destination array is columned-majored of size (max(NSHL * BS, stridey), batch_size)
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
__global__ void LoadElementValueKernel(index_type batch_size,
																			 const index_type* batch_index_ptr,
																			 const index_type* ien, int elem_size,
																			 const value_type* x, int stridex,
																			 value_type* y, int stridey,
																			 int NSHL) {

	index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= batch_size * NSHL) return;
	index_type i = idx / NSHL, lane = idx % NSHL;
	index_type iel = batch_index_ptr[i];
	index_type node_id = ien[iel * NSHL + lane];

	y += i * stridey;
	x += node_id * stridex;

	for (index_type j = 0; j < elem_size; ++j) {
		y[j * NSHL + lane] = x[j];
	}
}

/** Prefetch the value from the source array to the destination array belonging to the batch
 * The source array is columned-majored of size (max(elem_size, stride_src), num_node)
 * The destination array is columned-majored of size (max(NSHL * BS, stride_dst), batch_size)
 * @tparam NSHL The number of shape functions
 * @param[in] batch_size The number of elements in the batch
 * @param[in] batch_index_ptr[:] The index of the elements in the batch
 * @param[in] ien[NSHL, :] The index of the nodes in the element
 * @param[in] elem_size The size of the element
 * @param[in] src The source array to be loaded
 * @param[in] stride_src The increment of the source array, stride_src >= elem_size
 * @param[out] dst The destination array to be loaded
 * @param[in] stride_dst The increment of the destination array, stride_dst >= BS * NSHL
 */

__global__
void PrefetchElementValueKernel(index_type batch_size,
																index_type* batch_index_ptr,
																index_type* ien, int nshl,
																int elem_size,
																void* src, int stride_src,
																void* dst, int stride_dst) {
	index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= batch_size * nshl) return;
	index_type i = idx / nshl, lane = idx % nshl;
	index_type j;
	index_type iel = batch_index_ptr[i]; /* The index of the element in the batch */
	index_type node_id = ien[iel * nshl + lane]; /* The index of the node in the element */

	byte* src_ptr = (byte*)src;
	byte* dst_ptr = (byte*)dst;

	src_ptr += node_id * stride_src * elem_size;
	dst_ptr += i * stride_dst * elem_size;

	/* src_ptr[0:elem_size] -> dst_ptr[0:elem_size] */
	if(elem_size % sizeof(value_type) == 0) {
		for(j = 0; j < elem_size; j += sizeof(value_type)) {
			*(value_type*)(dst_ptr + j) = *(value_type*)(src_ptr + j);
		}
	}
	else {
		for(j = 0; j < elem_size; ++j) {
			dst_ptr[j] = src_ptr[j];
		}
	}
}

/** Add the elementwise residual vector to the global residual vector
 * This is the CUDA kernel for the function ElemRHSLocal2Global
 * @param[in]     batch_size The number of elements in the batch
 * @param[in]     batch_index_ptr[:] The index of the elements in the batch
 * @param[in]     ien[nshl, :] The index of the nodes in the element
 * @param[in]     blocklen The blocklen of the elementwise residual vector
 * @param[in]     elem_F[lda, nshl * batch_size] The elementwise residual vector
 * @param[in]     lda The leading dimension of the elem_F
 * @param[in,out] F[ldb, :] The global residual vector
 * @param[in]     ldb The block size
 */
__global__ void
ElemRHSLocal2GlobalKernel(index_type batch_size,
													index_type* batch_index_ptr, 
													index_type* ien, int nshl,
													index_type blocklen,
													value_type* elem_F, index_type lda,
													value_type* F, index_type ldb, index_type* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j, iel, node_id;
	if(i >= batch_size * nshl) {
		return;
	}
	if(mask && mask[i / nshl] == 0) {
		return;
	}
	iel = batch_index_ptr[i / nshl];
	node_id = ien[iel * nshl + i % nshl];
	elem_F += i * lda;
	F += node_id * ldb;

	for(j = 0; j < blocklen; ++j) {
		F[j] += elem_F[j];
	}
}

/** Get the normal vector of the element face
 * This is the CUDA kernel for the function GetElemFaceNV
 * @param[in] batch_size The number of elements in the batch
 * @param[in] metric[10 * batch_size] The metric tensor of the elements
 * @param[in] forn[batch_size] The index of the face orientation
 * @param[out] nv[3 * batch_size] The normal vector of the element face
 */
__global__ void GetElemFaceNVKernel(index_type batch_size, const value_type* metric, const index_type* forn, value_type* nv) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= batch_size) return;
	index_type iorn = forn[idx];
	
	nv += 3 * idx;
	/* metric (= metric + 10 * idx) is 3x3 and column-majored */
	metric += 10 * idx;
	value_type b[3] = {0.0, 0.0, 0.0};
	value_type detJ = metric[9];
	/* Using Nanson's formula */
	/* nv[0:3] = metric[0:3, 0:3] @ c_nv2[iorn, 0:3] */	
	for(index_type k = 0; k < 3; ++k) {
		for(index_type n = 0; n < 3; ++n) {
			b[n] += metric[n * 3 + k] * c_nv2[iorn * 3 + k];
		}
	}
	nv[0] = b[0] * detJ;
	nv[1] = b[1] * detJ;
	nv[2] = b[2] * detJ;
}


/** Get the elementwise determinant of the Jacobian matrix
 * This is the CUDA kernel for the function GetElemDetJ
 * @param[in]      ne The number of elements
 * @param[in, out] elem_metric[10 * ne] The metric tensor of the elements
 */
__global__ void GetElemDetJKernel(index_type ne, value_type* elem_metric) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= ne) return;

	value_type* elem_lu = elem_metric + idx * 10; /* elem_lu is the LU decomposition of the Jacobian matrix */
	elem_lu[9] = fabs(elem_lu[0] * elem_lu[4] * elem_lu[8]);
}

__global__ static void GetElemJ3DKernel(index_type batch_size, index_type* batch_index_ptr, index_type* ien,
																			  value_type* xg, value_type* elemJ) {
	const int NSHL = 4;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= batch_size) return;

	index_type iel = batch_index_ptr[idx];

	const value_type* xg_ptr0 = xg + ien[iel * NSHL + 0] * 3;
	const value_type* xg_ptr1 = xg + ien[iel * NSHL + 1] * 3;
	const value_type* xg_ptr2 = xg + ien[iel * NSHL + 2] * 3;
	const value_type* xg_ptr3 = xg + ien[iel * NSHL + 3] * 3;
	
	value_type* elemJ_ptr = elemJ + idx * 10;

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

__device__ void GetStabTau(const value_type* elem_G, const value_type* uadv, value_type rho, value_type cp, value_type mu, value_type kappa, value_type dt, value_type* tau) {
	const value_type* Ginv = elem_G;
	value_type tau_tmp[3] = {0.0, 0.0, 0.0};

	// /* Ginv = elem_invJ[0:3, 0:3] @ elem_invJ[0:3, 0:3].value_type */
	// #pragma unroll
	// for(index_type i = 0; i < 3; ++i) {
	// 	#pragma unroll
	// 	for(index_type j = 0; j < 3; ++j) {
	// 		#pragma unroll
	// 		for(index_type k = 0; k < 3; ++k) {
	// 			Ginv[i * 3 + j] += elem_invJ[i * 3 + k] * elem_invJ[j * 3 + k];
	// 		}
	// 	}
	// }

	/* tau_tmp[0] = 4.0 / dt^2 */
	tau_tmp[0] = 4.0 / (dt * dt);
	/* tau_tmp[1] = uadv[0:3] @ Ginv[0:3, 0:3] @ uadv[0:3] */
	/* tau_tmp[2] = Ginv[0:3, 0:3] @ Ginv[0:3, 0:3] */
	#pragma unroll
	for(index_type i = 0; i < 3; ++i) {
		#pragma unroll
		for(index_type j = 0; j < 3; ++j) {
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

__global__ void GetStridedBatchKernel(index_type batch_size, value_type* begin,
																			value_type** batch, index_type stride) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batch_size) return;
	batch[i] = begin + i * stride;
}


__device__ void GetrLi(const value_type* du, const value_type* uadv, const value_type* gradu, const value_type* gradp, value_type rho, value_type mu, value_type* rLi) {
	#pragma unroll
	for(index_type i = 0; i < 3; ++i) {
		rLi[i] = du[i] - fb[i] + uadv[0] * gradu[i*3+0] + uadv[1] * gradu[i*3+1] + uadv[2] * gradu[i*3+2];
		rLi[i] = rho * rLi[i] + gradp[i];
	}
}

__global__ void
AssembleWeakFormLHSKernel(index_type batch_size,
													const value_type* elem_G, const value_type* shgradg,
													const value_type* qr_wgalpha, const value_type* qr_dwgalpha, const value_type* qr_wggradalpha,
													value_type* fem_opt, value_type* elem_J) {
	const int NSHL = 4;
	index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
	index_type tx = threadIdx.x;
	index_type te, ti, lane;
	index_type aa = (idx % (NSHL * NSHL)) / NSHL;
	index_type bb = (idx % (NSHL * NSHL)) % NSHL;

	index_type iel = idx / (NSHL * NSHL);


	index_type BLK_ELEM_COUNT_MAX = blockDim.x / (NSHL * NSHL);
	index_type BLK_ELEM_BEGIN = blockIdx.x * blockDim.x / (NSHL * NSHL);
	index_type BLK_ELEM_COUNT = (BLK_ELEM_COUNT_MAX < batch_size - BLK_ELEM_BEGIN ? BLK_ELEM_COUNT_MAX : batch_size - BLK_ELEM_BEGIN);


	value_type kDT = fem_opt[OPTION_DT];
	value_type kALPHAM = fem_opt[OPTION_ALPHA_M];
	value_type kALPHAF = fem_opt[OPTION_ALPHA_F];
	value_type kGAMMA = fem_opt[OPTION_GAMMA];

	int BS = (int)(4 * fem_opt[OPTION_NS] + fem_opt[OPTION_T] + fem_opt[OPTION_PHI]);

	const f64 fact1 = kALPHAM;
	const f64 fact2 = kDT * kALPHAF * kGAMMA;

	// __shared__ value_type sh_buff[BLK_ELEM_COUNT_MAX * (NSHL * NSHL + 4)];
	extern __shared__ value_type sh_buff[];

	typedef value_type (T12)[NSHL * 3];
	typedef value_type (T4)[NSHL];
	T12* shgradl = (T12*)sh_buff;
	T4* shconv = (T4*)(sh_buff + BLK_ELEM_COUNT_MAX * NSHL * 3);
	T4* tau_component = (T4*)(sh_buff + BLK_ELEM_COUNT_MAX * NSHL * NSHL);

	if(tx < BLK_ELEM_COUNT * NSHL * 3) {
		((value_type*)shgradl)[tx] = shgradg[BLK_ELEM_BEGIN * NSHL * 3 + tx];
	}
	if(tx < BLK_ELEM_COUNT) {
		value_type gg = 0.0, tr = 0.0, gij;
		for(index_type i = 0; i < 9; ++i) {
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


	// for(index_type i = 0; i < NSHL * NSHL * BS * BS; ++i) {
	// 	elem_J[i] = 0.0;
	// }

	value_type tau[4] = {0.0, 0.0, 0.0, 0.0}; 
	// value_type divu = qr_wggradalpha[iel * 3 * BS + 0]
	// 				+ qr_wggradalpha[iel * 3 * BS + 4]
	// 				+ qr_wggradalpha[iel * 3 * BS + 8];
	
	// value_type* elem_J_buffer = elem_J + (iel * NSHL * NSHL + aa * NSHL + bb) * BS * BS;
	if(iel >= batch_size) return;

	value_type elem_J_buffer[4*4];
	for(index_type i = 0; i < 4*4; ++i) {
		elem_J_buffer[i] = 0.0;
	}

	value_type tmp;
	value_type detJ = elem_G[iel * 10 + 9];
	const value_type knu = kMU / kRHO;
	const value_type kalpha = kKAPPA / kRHO / kCP;
	for(index_type iq = 0; iq < NQR(gw); ++iq) {
		/* Update shconv[:][:] */
		if(tx < BLK_ELEM_COUNT * NSHL) {
			te = tx / NSHL;
			lane = tx % NSHL;
			/* tx = te * NSHL + lane */
			/* shconv[te][lane] = shgradl[te][lane][0:3] * uadv[0:3] */
			shconv[te][lane] = 0.0;
			shconv[te][lane] += shgradl[te][lane * 3 + 0] * qr_wgalpha[NQR(gw) * BS * (BLK_ELEM_BEGIN + te) + 0 * NQR(gw) + iq];
			shconv[te][lane] += shgradl[te][lane * 3 + 1] * qr_wgalpha[NQR(gw) * BS * (BLK_ELEM_BEGIN + te) + 1 * NQR(gw) + iq];
			shconv[te][lane] += shgradl[te][lane * 3 + 2] * qr_wgalpha[NQR(gw) * BS * (BLK_ELEM_BEGIN + te) + 2 * NQR(gw) + iq];
		}
		__syncthreads();
		/* Get tau */
		/* Update tau_component[:][0] */
		if(tx < BLK_ELEM_COUNT) {
			// value_type uadv[3];
			// uadv[0] = qr_wgalpha[NQR * BS * (BLK_ELEM_BEGIN + tx) + 0 * NQR + iq];
			// uadv[1] = qr_wgalpha[NQR * BS * (BLK_ELEM_BEGIN + tx) + 1 * NQR + iq];
			// uadv[2] = qr_wgalpha[NQR * BS * (BLK_ELEM_BEGIN + tx) + 2 * NQR + iq];
			tmp = 0;
			tmp += shconv[tx][1] * shconv[tx][1];
			tmp += shconv[tx][2] * shconv[tx][2];
			tmp += shconv[tx][3] * shconv[tx][3];
			// for(index_type i = 0; i < 3; ++i) {
			// 	for(index_type j = 0; j < 3; ++j) {
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

		value_type eK = shgradl[te][aa * 3 + 0] * shgradl[te][bb * 3 + 0] +
						shgradl[te][aa * 3 + 1] * shgradl[te][bb * 3 + 1] +
						shgradl[te][aa * 3 + 2] * shgradl[te][bb * 3 + 2];
		value_type detJgw = detJ * gw[iq];
		tmp = 0.0;
		tmp += fact1 * kRHO * shlu[aa * NQR(gw) + iq] * shlu[bb * NQR(gw) + iq];
		tmp += fact1 * kRHO * kRHO * tau[0] * shconv[te][aa] * shlu[bb * NQR(gw) + iq];
		tmp += fact2 * shlu[aa * NQR(gw) + iq] * kRHO * shconv[te][bb];
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
		for(index_type ii = 0; ii < 3; ++ii) {
			for(index_type jj = 0; jj < 3; ++jj) {
				elem_J_buffer[ii * 4 + jj] += fact2 * kMU * shgradl[te][aa * 3 + jj] * shgradl[te][bb * 3 + ii] * detJgw;
				elem_J_buffer[ii * 4 + jj] += fact2 * kRHO * tau[1] * shgradl[te][aa * 3 + ii] * shgradl[te][bb * 3 + jj] * detJgw;
			}
		}

		/* dRM/dP */
		for(index_type ii = 0; ii < 3; ++ii) {
			elem_J_buffer[ii * 4 + 3] -= shgradl[te][aa * 3 + ii] * shlu[bb * NQR(gw) + iq] * detJgw;
			elem_J_buffer[ii * 4 + 3] += kRHO * tau[0] * shconv[te][aa] * shgradl[te][bb * 3 + ii] * detJgw;
		}

		// tau[0] = 0.0;
		/* dRC/dU */
		for(index_type ii = 0; ii < 3; ++ii) {
			elem_J_buffer[3 * 4 + ii] += fact1 * kRHO * tau[0] * shgradl[te][aa * 3 + ii] * shlu[bb * NQR(gw) + iq] * detJgw;
			elem_J_buffer[3 * 4 + ii] += fact2 * shlu[aa * NQR(gw) + iq] * shgradl[te][bb * 3 + ii] * detJgw;
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
		value_type* buff = sh_buff;
		buff[tx] = 0.0;

		index_type THREAD_STRIDE = 8; /* Hard coded for 256 threads and 24 leading dimensions */
		index_type wrap_id = tx / 32;
		lane = tx % 32;
		for(index_type e = 0; e < BLK_ELEM_COUNT * 2; ++e) {
			/* Load elem_J_buffer into buff */
			/* The thread range is [e * THREAD_STRIDE, (e + 1) * THREAD_STRIDE) */ 
			index_type tx_start = e * THREAD_STRIDE;
			index_type tx_end = tx_start + THREAD_STRIDE;
			if(tx_start <= tx && tx < tx_end) {
				#pragma unroll
				for(index_type i = 0; i < 4; i++) {
					#pragma unroll
					for(index_type j = 0; j < 4; j++) {
						buff[(tx - tx_start) * 24 + i * 6 + j] = elem_J_buffer[i * 4 + j];
					}
				}
			}
			__syncthreads();

			/* Update elem_J */
			index_type e_idx = BLK_ELEM_BEGIN + e / 2;
			value_type* dst = elem_J + (e_idx * NSHL * NSHL + wrap_id + (e & 1) * THREAD_STRIDE) * BS * BS;
			if(lane < 24) {
				dst[lane] = buff[tx];
			}
		}
	}
	else {
		elem_J += (iel * NSHL * NSHL + aa * NSHL + bb) * BS * BS;
		#pragma unroll
		for(index_type i = 0; i < 4; ++i) {
			#pragma unroll
			for(index_type j = 0; j < 4; ++j) {
				// elem_J[i * BS + j] += elem_J_buffer[i * 4 + j];
				atomicAdd(elem_J + i * BS + j, elem_J_buffer[i * 4 + j]);
			}
		}
	}
	elem_J[M2D(4, 4)] += aa == bb;
	elem_J[M2D(5, 5)] += aa == bb; 
}

__global__ void
AssembleWeakFormRHSKernel(index_type batch_size,
												  value_type* elem_G, value_type* shgradg,
												  value_type* qr_wgalpha,
													value_type* qr_dwgalpha,
													value_type* qr_wggradalpha,
													value_type* fem_opt,
												  value_type* elem_F) {
	const int NSHL = 4;
	index_type iel = blockIdx.x * blockDim.x + threadIdx.x;
	if(iel >= batch_size) return;

	value_type kDT = fem_opt[OPTION_DT];
	value_type kALPHAM = fem_opt[OPTION_ALPHA_M];
	value_type kALPHAF = fem_opt[OPTION_ALPHA_F];
	value_type kGAMMA = fem_opt[OPTION_GAMMA];

	int BS = (int)(4 * fem_opt[OPTION_NS] + fem_opt[OPTION_T] + fem_opt[OPTION_PHI]);

	elem_G += iel * 10;
	value_type detJ = elem_G[9];
	shgradg += iel * NSHL * 3;
	/* Column-majored */
	/* [:, 0:3]: Velocity */
	/* [:, 3]: Pressure */
	/* [:, 4]: Phi */
	/* [:, 5]: Temperature */
	qr_wgalpha += iel * NQR(gw) * BS; /* (NQR, BS) */
	qr_dwgalpha += iel * NQR(gw) * BS; /* (NQR, BS) */
	qr_wggradalpha += iel * 3 * BS; /* (3, BS) */

	elem_F += iel * NSHL * BS;
	for(index_type i = 0; i < NSHL * BS; ++i) {
		elem_F[i] = 0.0;
	}


	value_type tau[4] = {0.0, 0.0, 0.0, 0.0}; 
	value_type divu = qr_wggradalpha[0] + qr_wggradalpha[4] + qr_wggradalpha[8];
	
	value_type rLi[3];
	value_type uadv[3];
	value_type shconv[NSHL];

	for(index_type iq = 0; iq < NQR(gw); ++iq) {
		/* Get uadv */
		uadv[0] = qr_wgalpha[NQR(gw) * 0 + iq];
		uadv[1] = qr_wgalpha[NQR(gw) * 1 + iq];
		uadv[2] = qr_wgalpha[NQR(gw) * 2 + iq];

		/* Get rLi */
		for(index_type i = 0; i < 3; ++i) {
			rLi[i] = 0.0;
			rLi[i] += kRHO * (qr_dwgalpha[NQR(gw) * i + iq] - fb[i]);
			rLi[i] += kRHO * uadv[0] * qr_wggradalpha[3 * i + 0];
			rLi[i] += kRHO * uadv[1] * qr_wggradalpha[3 * i + 1];
			rLi[i] += kRHO * uadv[2] * qr_wggradalpha[3 * i + 2];
			rLi[i] += qr_wggradalpha[3 * 3 + i];
		}

		/* Get Stab */
		GetStabTau(elem_G, uadv, kRHO, kCP, kMU, kKAPPA, kDT, tau);

		for(index_type aa = 0; aa < NSHL; ++aa) {
			// shconv[aa] = uadv[0] * shgradg[aa * 3 + 0] + uadv[1] * shgradg[aa * 3 + 1] + uadv[2] * shgradg[aa * 3 + 2];
			shconv[aa] = 0.0;
			shconv[aa] += uadv[0] * shgradg[aa * 3 + 0];
			shconv[aa] += uadv[1] * shgradg[aa * 3 + 1];
			shconv[aa] += uadv[2] * shgradg[aa * 3 + 2];
		}

		value_type tmp0[3];
		value_type tmp1[3 * 3];
		/* Get tmp0 */
		for(index_type i = 0; i < 3; ++i) {
			tmp0[i] = 0.0;
			tmp0[i] += kRHO * (qr_dwgalpha[NQR(gw) * i + iq] - fb[i]);
			tmp0[i] += kRHO * (uadv[0] - tau[0] * rLi[0]) * qr_wggradalpha[3 * i + 0];
			tmp0[i] += kRHO * (uadv[1] - tau[0] * rLi[1]) * qr_wggradalpha[3 * i + 1];
			tmp0[i] += kRHO * (uadv[2] - tau[0] * rLi[2]) * qr_wggradalpha[3 * i + 2];
		}

		/* Get tmp1 */
		for(index_type i = 0; i < 3; ++i) {
			for(index_type j = 0; j < 3; ++j) {
				tmp1[i * 3 + j] = 0.0;
				tmp1[i * 3 + j] += kMU * (qr_wggradalpha[3 * i + j] + qr_wggradalpha[3 * j + i]);
				tmp1[i * 3 + j] += kRHO * tau[0] * rLi[i] * uadv[j];
				tmp1[i * 3 + j] -= kRHO * tau[0] * tau[0] * rLi[i] * rLi[j];
			}
		}
		for(index_type i = 0; i < 3; ++i) {
			tmp1[i * 3 + i] += -qr_wgalpha[NQR(gw) * 3 + iq] + kRHO * tau[1] * divu;
		}

		/* Get tmp2 */
		// tmp2[0] = -qr_wgalpha[NQR * 3 + iq] + kRHO * tau[1] * divu;

		/* Momentum */
		#pragma unroll
		for(index_type aa = 0; aa < NSHL; ++aa) {
			#pragma unroll
			for(index_type ii = 0; ii < 3; ++ii) {
				value_type bm = 0.0;
				bm += shlu[aa * NQR(gw) + iq] * tmp0[ii];
				bm += shgradg[aa * 3 + 0] * tmp1[ii * 3 + 0];
				bm += shgradg[aa * 3 + 1] * tmp1[ii * 3 + 1];
				bm += shgradg[aa * 3 + 2] * tmp1[ii * 3 + 2];
				// bm += shgradg[aa * 3 + ii] * tmp2[0];
				elem_F[M2D(aa, ii)] += bm * gw[iq] * detJ;
			}
		}
		/* Continuity */
		#pragma unroll
		for(index_type aa = 0; aa < NSHL; ++aa) {
			value_type bc = 0.0;
			bc += shlu[aa * NQR(gw) + iq] * divu;
			bc += tau[0] * rLi[0] * shgradg[aa * 3 + 0];
			bc += tau[0] * rLi[1] * shgradg[aa * 3 + 1];
			bc += tau[0] * rLi[2] * shgradg[aa * 3 + 2];
			elem_F[M2D(aa, 3)] += bc * gw[iq] * detJ;
		}
		/* Phi */
		for(index_type aa = 0; aa < NSHL; ++aa) {
			value_type bp = qr_dwgalpha[NQR(gw) * 4 + iq]
					 + uadv[0] * qr_wggradalpha[3 * 4 + 0]
					 + uadv[1] * qr_wggradalpha[3 * 4 + 1]
					 + uadv[2] * qr_wggradalpha[3 * 4 + 2];
			elem_F[M2D(aa, 4)] += bp * (shlu[aa * NQR(gw) + iq] + tau[2] * shconv[aa]) * gw[iq] * detJ;
		}

		/* Temperature */
		for(index_type aa = 0; aa < NSHL; ++aa) {
			value_type bt = kRHO * kCP * (qr_dwgalpha[NQR(gw) * 5 + iq]
					 + uadv[0] * qr_wggradalpha[3 * 5 + 0]
					 + uadv[1] * qr_wggradalpha[3 * 5 + 1]
					 + uadv[2] * qr_wggradalpha[3 * 5 + 2]) 
					 * (shlu[aa * NQR(gw) + iq] + kRHO * kCP * tau[3] * shconv[aa]);

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
	

}

__global__ void
FaceAssemblyKernel(index_type batch_size, const value_type* elem_invJ, const value_type* nv, const value_type* shgradg,
									 const index_type* forn, const value_type* qr_wgalpha, const value_type* qr_wggradalpha,
									 value_type* fem_opt, value_type* elem_F, value_type* elem_J) {
	const int NSHL = 4;
	index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= batch_size) return;

	value_type kDT = fem_opt[OPTION_DT];
	value_type kALPHAM = fem_opt[OPTION_ALPHA_M];
	value_type kALPHAF = fem_opt[OPTION_ALPHA_F];
	value_type kGAMMA = fem_opt[OPTION_GAMMA];

	int BS = (int)(4 * fem_opt[OPTION_NS] + fem_opt[OPTION_T] + fem_opt[OPTION_PHI]);

	index_type iorn = forn[idx];
	elem_invJ += idx * 10;
	nv += idx * 3;
	shgradg += idx * 3 * NSHL;
	qr_wgalpha += idx * NQR(gwb) * BS;
	qr_wggradalpha += idx * 3 * BS;


	value_type hinv = 0.0, detJb = 0.0;
	value_type uadv[3];
	for(index_type i = 0; i < 3; ++i) {
		uadv[i] = elem_invJ[i + 3 * 0] * nv[0] + elem_invJ[i + 3 * 1] * nv[1] + elem_invJ[i + 3 * 2] * nv[2];
		hinv += uadv[i] * uadv[i];
		detJb += nv[i] * nv[i];
	}
	detJb = sqrt(detJb);
	// hinv = sqrt(hinv) / detJb;
	hinv = sqrt(hinv);
	value_type tau_b = 4.0 * kMU * hinv;
	// tau_b = 4.0 * kMU / 0.01 * detJb;

	/* Assemble the weak imposition of the boundary condition */
	if(elem_F) {
		elem_F += idx * NSHL * BS;
		for(index_type i = 0; i < NSHL * BS; ++i) {
			elem_F[i] = 0.0;
		}
		value_type tmp0[3];
		value_type tmp1[3 * 3];
		for(index_type iq = 0; iq < NQR(gwb); ++iq) {
			uadv[0] = qr_wgalpha[NQR(gwb) * 0 + iq];
			uadv[1] = qr_wgalpha[NQR(gwb) * 1 + iq];
			uadv[2] = qr_wgalpha[NQR(gwb) * 2 + iq];
			value_type unor = uadv[0] * nv[0] + uadv[1] * nv[1] + uadv[2] * nv[2];
			value_type uneg = (unor - fabs(unor)) * 0.5;
			for(index_type i = 0; i < 3; ++i) {
				tmp0[i] = 0.0;
				tmp0[i] += nv[i] * qr_wgalpha[NQR(gwb) * 3 + iq];
				tmp0[i] -= kMU * (nv[0] * qr_wggradalpha[3 * i + 0] +
													nv[1] * qr_wggradalpha[3 * i + 1] +
													nv[2] * qr_wggradalpha[3 * i + 2]);
				tmp0[i] -= kMU * (nv[0] * qr_wggradalpha[3 * 0 + i] +
													nv[1] * qr_wggradalpha[3 * 1 + i] +
													nv[2] * qr_wggradalpha[3 * 2 + i]);
				tmp0[i] -= kRHO * uneg * uadv[i]; // qr_wgalpha[NQR(gwb) * i + iq];
				tmp0[i] += tau_b * uadv[i]; // qr_wgalpha[NQR(gwb) * i + iq];

			}	

			for(index_type i = 0; i < 3; ++i) {
				for(index_type j = 0; j < 3; ++j) {
					// tmp1[i * 3 + j] = -kMU * (nv[i] * qr_wgalpha[NQR(gwb) * j + iq] + nv[j] * qr_wgalpha[NQR(gwb) * i + iq]);
					tmp1[i * 3 + j] = -kMU * (nv[i] * uadv[j] + nv[j] * uadv[i]);
				}
			}

			for(index_type aa = 0; aa < NSHL; ++aa) {
				for(index_type ii = 0; ii < 3; ++ii) {
					value_type bm = 0.0;
					// bm += c_shlub[NQR(gwb) * NSHL * iorn + iq * NSHL + aa] * tmp0[ii];
					bm += shlub[iorn][iq * NSHL + aa] * tmp0[ii];
					bm += shgradg[aa * 3 + 0] * tmp1[ii * 3 + 0];
					bm += shgradg[aa * 3 + 1] * tmp1[ii * 3 + 1];
					bm += shgradg[aa * 3 + 2] * tmp1[ii * 3 + 2];
					elem_F[M2D(aa, ii)] += bm * gwb[iq];
				}
				// elem_F[M2D(aa, 3)] -= c_shlub[NQR(gwb) * NSHL * iorn + iq * NSHL + aa] * unor * gwb[iq];
				elem_F[M2D(aa, 3)] -= shlub[iorn][iq * NSHL + aa] * unor * gwb[iq];
				// elem_F[M2D(aa, 3)] = 0.0;
				// for(index_type ii = 0; ii < 4; ++ii) {
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
		value_type fact1 = kALPHAM;
		value_type fact2 = kDT * kALPHAF * kGAMMA;
		value_type tmp0, tmp1;
		elem_J += idx * NSHL * NSHL * BS * BS;
		for(index_type i = 0; i < NSHL * NSHL * BS * BS; ++i) {
			elem_J[i] = 0.0;
		}
		value_type shnorm[NSHL];
		for(index_type aa = 0; aa < NSHL; ++aa) {
			shnorm[aa] = 0.0;
			shnorm[aa] += shgradg[aa * 3 + 0] * nv[0];
			shnorm[aa] += shgradg[aa * 3 + 1] * nv[1];
			shnorm[aa] += shgradg[aa * 3 + 2] * nv[2];
		}

		for(index_type iq = 0; iq < NQR(gwb); ++iq) {
			uadv[0] = qr_wgalpha[NQR(gwb) * 0 + iq];
			uadv[1] = qr_wgalpha[NQR(gwb) * 1 + iq];
			uadv[2] = qr_wgalpha[NQR(gwb) * 2 + iq];
			value_type unor = uadv[0] * nv[0] + uadv[1] * nv[1] + uadv[2] * nv[2];
			value_type uneg = (unor - fabs(unor)) * 0.5;
			for(index_type aa = 0; aa < NSHL; ++aa) {
				for(index_type bb = 0; bb < NSHL; ++bb) {
					/* dRM/dU */
					tmp0 = 0.0;
					// tmp0 -= kMU * (shnorm[bb] * c_shlub[NQR(gwb) * NSHL * iorn + iq * NSHL + aa] 
					// 						+ shnorm[aa] * c_shlub[NQR(gwb) * NSHL * iorn + iq * NSHL + bb]);
					tmp0 -= kMU * (shnorm[bb] * shlub[iorn][iq * NSHL + aa] 
											+ shnorm[aa] * shlub[iorn][iq * NSHL + bb]);
					// tmp0 -= kRHO * c_shlub[NQR(gwb) * NSHL * iorn + iq * NSHL + aa] 
					// 						 * c_shlub[NQR(gwb) * NSHL * iorn + iq * NSHL + bb] * uneg;
					tmp0 -= kRHO * shlub[iorn][iq * NSHL + aa] 
										 * shlub[iorn][iq * NSHL + bb] * uneg;
					// tmp0 += tau_b * c_shlub[NQR(gwb) * NSHL * iorn + iq * NSHL + aa] * c_shlub[NQR(gwb) * NSHL * iorn + iq * NSHL + bb];
					tmp0 += tau_b * shlub[iorn][iq * NSHL + aa] * shlub[iorn][iq * NSHL + bb];
					elem_J[M4D(aa, bb, 0, 0)] += fact2 * tmp0 * gwb[iq];
					elem_J[M4D(aa, bb, 1, 1)] += fact2 * tmp0 * gwb[iq];
					elem_J[M4D(aa, bb, 2, 2)] += fact2 * tmp0 * gwb[iq];
					// if(idx == 0 && aa == 0 && bb == 0) {
					// 		printf("sys = %.17e tauB= %.17e, uneg=%.17e, unor=%.17e\n", tmp0, tau_b, uneg, unor);
					// }
					
					for(index_type ii = 0; ii < 3; ++ii) {
						for(index_type jj = 0; jj < 3; ++jj) {
							tmp0 = 0.0;
							// tmp0 -= kMU * c_shlub[NQR(gwb) * NSHL * iorn + iq * NSHL + aa] * shgradg[bb * 3 + ii] * nv[jj];
							// tmp0 -= kMU * c_shlub[NQR(gwb) * NSHL * iorn + iq * NSHL + bb] * shgradg[aa * 3 + jj] * nv[ii];
							tmp0 -= kMU * shlub[iorn][iq * NSHL + aa] * shgradg[bb * 3 + ii] * nv[jj];
							tmp0 -= kMU * shlub[iorn][iq * NSHL + bb] * shgradg[aa * 3 + jj] * nv[ii];
							// if(idx == 0 && aa == 0 && bb == 0) {
							// 	printf("sys[%d][%d] = %.17e %.17e %.17e\n", ii, jj, fact2, tmp0, gwb[iq]);
							// }
							elem_J[M4D(aa, bb, ii, jj)] += fact2 * tmp0 * gwb[iq];
						}
					}

					// tmp0 = c_shlub[NQR(gwb) * NSHL * iorn + iq * NSHL + aa] * c_shlub[NQR(gwb) * NSHL * iorn + iq * NSHL + bb];
					tmp0 = shlub[iorn][iq * NSHL + aa] * shlub[iorn][iq * NSHL + bb];
					for(index_type ii = 0; ii < 3; ++ii) {
						/* dRC/dU */
						elem_J[M4D(aa, bb, 3, ii)] -= fact2 * tmp0 * nv[ii] * gwb[iq]; 
						/* dRM/dP */
						elem_J[M4D(aa, bb, ii, 3)] += tmp0 * nv[ii] * gwb[iq];
					}

				}
			}
		}
#ifdef DBG_TET
		if(idx == 0 && elem_J) {
			index_type aa = 0;
			index_type bb = 0;
			printf("num_thread = %d\n", blockDim.x * gridDim.x);
			printf("batch_size = %d\n", batch_size);
			printf("uadv=[%e %e %e]\n", uadv[0], uadv[1], uadv[2]);
			printf("nv=[%e %e %e]\n", nv[0], nv[1], nv[2]);
			printf("elem_J[M4D(%d, %d, :, :)] = \n", aa, bb);
			for(index_type ii = 0; ii < 4; ++ii) {
				for(index_type jj = 0; jj < 4; ++jj) {
					printf("%.17e ", elem_J[M4D(aa, bb, ii, jj)]);
				}
				printf("\n");
			}

		}
#endif

	}
}

__global__ void
GetShapeGradKernel(index_type num_elem, const value_type* elem_invJ, value_type* shgrad) {
	i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num_elem) return;

	const value_type* elem_invJ_ptr = elem_invJ + idx * 10;
	value_type* shgrad_ptr = shgrad + idx * 12;
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



void GetShapeGrad(index_type num_elem, value_type* elem_invJ, value_type* shgrad) {
	int block_size = 256;
	int num_block = CEIL_DIV(num_elem, block_size);
	GetShapeGradKernel<<<num_block, block_size>>>(num_elem, elem_invJ, shgrad);
}

void GetElemInvJ3D(index_type batch_size, index_type* batch_index_ptr,
									 index_type* ien, value_type* coord, value_type* elem_metric, Arena scratch) {
	value_type one = 1.0, zero = 0.0;
	int block_size = 256;
	int num_block = CEIL_DIV(batch_size, block_size);

	/* Load the elements in the batch into elem_metric[0:10, :]  */
	GetElemJ3DKernel<<<num_block, block_size>>>(batch_size, batch_index_ptr, ien, coord, elem_metric);

	/* Reverse the matrices */
	value_type* d_elem_buff = (value_type*)ArenaPush(sizeof(value_type), batch_size * 10, &scratch, 0);
	value_type** d_mat_batch = (value_type**)ArenaPush(sizeof(value_type*), batch_size, &scratch, 0);
	value_type** d_matinv_batch = (value_type**)ArenaPush(sizeof(value_type*), batch_size, &scratch, 0);
	int* info = (int*)ArenaPush(sizeof(int), batch_size, &scratch, 0);	
	int* pivot = (int*)ArenaPush(sizeof(int), batch_size * 3, &scratch, 0);

	/* Set mat_batch */
	GetStridedBatchKernel<<<num_block, block_size>>>(batch_size, d_elem_buff, d_matinv_batch, 10);
	GetStridedBatchKernel<<<num_block, block_size>>>(batch_size, elem_metric, d_mat_batch, 10);

	/* LU decomposition */
	BLAS_CALL(getrfBatched, 3, d_mat_batch, 3, pivot, info, batch_size);
	/* Calculate the determinant of the Jacobian matrix */
	GetElemDetJKernel<<<num_block, block_size>>>(batch_size, elem_metric);
	/* Invert the Jacobian matrix */
	BLAS_CALL(getriBatched, 3, d_mat_batch, 3, pivot, d_matinv_batch, 3, info, batch_size);
	/* Copy the inverse of the Jacobian matrix to the elem_metric */
	BLAS_CALL(geam, BLAS_N, BLAS_N,
						9, batch_size,
						&one,
						d_elem_buff, 10,
						&zero,
						elem_metric, 10,
						elem_metric, 10);

}


void FaceAssembly(index_type batch_size, const value_type* elem_invJ, const value_type* nv, const value_type* shgradg,
									const index_type* forn, const value_type* qr_wgalpha, const value_type* qr_wggradalpha,
									FEMOptions opt, value_type* elem_F, value_type* elem_J) {
	i32 block_dim = 256;
	i32 grid_dim = CEIL_DIV(batch_size, block_dim);
	const int NSHL = 4;
	value_type* h_opt_pinned = NULL;
	cudaMallocHost(&h_opt_pinned, sizeof(FEMOptions));
	CdamMemcpy(h_opt_pinned, &opt, sizeof(FEMOptions), HOST_MEM, HOST_MEM);
	FaceAssemblyKernel<<<grid_dim, block_dim>>>(batch_size, elem_invJ, nv, shgradg, forn, qr_wgalpha, qr_wggradalpha, h_opt_pinned, elem_F, elem_J);
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
	cudaFreeHost(h_opt_pinned);
}

void IntElemAssembly(index_type batch_size, value_type* elem_G, value_type* shgradg,
										 value_type* qr_wgalpha, value_type* qr_dwgalpha, value_type* qr_wggradalpha,
										 FEMOptions opt,
										 value_type* elem_F, value_type* elem_J) {
	const int NSHL = 4;
	int num_thread, num_block;
	value_type* h_opt_pinned = NULL;
	cudaMallocHost(&h_opt_pinned, sizeof(FEMOptions));
	CdamMemcpy(h_opt_pinned, &opt, sizeof(FEMOptions), HOST_MEM, HOST_MEM);
	if(elem_F) {
		num_thread = 256;
		num_block = CEIL_DIV(batch_size, num_thread);
		AssembleWeakFormRHSKernel<<<num_block, num_thread>>>(
				batch_size,
				elem_G, shgradg,
				qr_wgalpha, qr_dwgalpha, qr_wggradalpha,
				h_opt_pinned, elem_F);	
	}

	if(elem_J) {
		num_thread = 256;
		num_block = CEIL_DIV(batch_size * NSHL * NSHL, num_thread);
		int extern_shared_mem_size = num_thread / (NSHL * NSHL) * (NSHL * NSHL + 4);
		AssembleWeakFormLHSKernel<<<num_block, num_thread, extern_shared_mem_size>>>(
				batch_size,
				elem_G, shgradg,
				qr_wgalpha, qr_dwgalpha, qr_wggradalpha,
				h_opt_pinned, elem_J);
	}

	cudaFreeHost(h_opt_pinned);
}

#endif /* CDAM_USE_CUDA */

__END_DECLS__


