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

template<typename I, typename T, I bs=1, I NSHL=4>
__global__ void LoadElementValueKernel(I n, const I* ien, const I* index, const T* src, T* dst) {
	I idx = blockIdx.x * blockDim.x + threadIdx.x;
	I num_thread = blockDim.x * gridDim.x;
	if(idx >= n) return;
	I iel = index[idx];

	#pragma unroll
	for(I i = 0; i < NSHL; ++i) {
		I node_id = ien[iel * NSHL + i];
		#pragma unroll
		for(I j = 0; j < bs; ++j) {
			dst[(idx * NSHL + i) * bs + j] = src[node_id * bs + j];
		}
	}
	/*
	while(idx < n) {
		#pragma unroll
		for(I i = 0; i < NSHL; ++i) {
			I node_id = ien[index[idx] * NSHL + i];
			#pragma unroll
			for(I j = 0; j < bs; ++j) {
				dst[idx * bs + j] = src[node_id * bs + j];
			}
		}
		idx += num_thread;
	}
	*/
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


template <typename I, typename T, I bs=1, I NSHL=4>
__global__ void
ElemRHSLocal2GlobalKernel(I batch_size, const I* batch_index_ptr, const I* ien, const T* elem_F, T* F) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batch_size) {
		return;
	}
	int iel = batch_index_ptr[i];
	ien += iel * NSHL;
	I vert[NSHL] = {ien[0], ien[1], ien[2], ien[3]};
	#pragma unroll
	for(int j = 0; j < NSHL; ++j) {
		int vertex_id = vert[j];
		#pragma unroll
		for(int k = 0; k < bs; ++k) {
			F[vertex_id * bs + k] += elem_F[(i * NSHL + j) * bs + k];
		}
	}
	// for(int j = i; j < num_elem * NSHL; j += blockDim.x * gridDim.x) {
	// 	int vertex_id = ien[iel[j / NSHL] * NSHL + j % NSHL];
	// 	#pragma unroll
	// 	for(int j = 0; j < bs; ++j) {
	// 		F[vertex_id * bs + j] += elem_F[ * bs + j];
	// 	}
	// }
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
AssemleWeakFormKernel(I batch_size, const T* elem_invJ,
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
__global__ void GetElemInvJ3DLoadMatrixBatchKernel(I batch_size, T** A, T* elem_metric) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batch_size) return;
	A[i] = elem_metric + i * 10;
}

template <typename I, typename T, typename INC>
__global__ void IncrementByN(I n, T* a, INC inc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n) return;
	a[i] += inc;
}

/**
 * @brief Calculate the inverse of the Jacobian matrix of the element
 * @param[in] ne   The number of elements
 * @param[in] ien  The index of the element in the mesh
 * @param[in] xg   The global coordinates of the nodes
 * @param[out] elem_metric The inverse of the Jacobian matrix of the element
 */
template<typename I, typename T, I NSHL=4>
void GetElemInvJ3D(I batch_size, const I* ien, const I* batch_index_ptr, const T* xg, T* elem_metric) {
	f64 one = 1.0, zero = 0.0;
	int block_size = 256;
	int num_block = (batch_size + block_size - 1) / block_size;
	CUGUARD(cudaGetLastError());

	cudaStream_t stream[1];	
	cudaStreamCreate(stream + 0);

	cublasHandle_t handle;
	cublasCreate(&handle);

	/* 0. Load the elements in the batch into elem_metric[0:10, :]  */
	GetElemJ3DKernel<I, T, NSHL><<<num_block, block_size>>>(batch_size, ien, batch_index_ptr, xg, elem_metric);



	const int MAX_BATCH_SIZE = 128*8;
	static int* info = NULL;
	if(!info) {
		info = (int*)CdamMallocDevice(MAX_BATCH_SIZE * sizeof(int) * 4);
	}
	int* pivot = info + MAX_BATCH_SIZE;

	static T* d_elem_buff = NULL;
	if(!d_elem_buff) {
		d_elem_buff = (T*)CdamMallocDevice(MAX_BATCH_SIZE * 10 * sizeof(T));
	}
	static T** d_mat_batch = NULL;
	if (!d_mat_batch)
		d_mat_batch = (T**)CdamMallocDevice(MAX_BATCH_SIZE * sizeof(T*) * 2);
	T** d_matinv_batch = d_mat_batch + MAX_BATCH_SIZE;

	/* Set mat_batch */
	thrust::transform(thrust::device,
										thrust::make_counting_iterator<u32>(0),
										thrust::make_counting_iterator<u32>(MAX_BATCH_SIZE),
										d_matinv_batch,
										thrust::placeholders::_1 * 10 + d_elem_buff);
	// thrust::transform(thrust::device,
	// 									thrust::make_counting_iterator<u32>(0),
	// 									thrust::make_counting_iterator<u32>(MAX_BATCH_SIZE),
	// 									d_mat_batch,
	// 									thrust::placeholders::_1 * 10 + elem_metric);

	cublasSetStream(handle, stream[0]);

	for(int i = 0; i < batch_size; i += MAX_BATCH_SIZE) {
		int num_elem = (i + MAX_BATCH_SIZE < batch_size) ? MAX_BATCH_SIZE : batch_size - i;

		// printf("ne[%d] = %d out of %d\n", i/MAX_BATCH_SIZE, num_elem, batch_size);

		/* Set matinv_batch */
		// thrust::transform(thrust::device,
		// 									thrust::make_counting_iterator<u32>(0),
		// 									thrust::make_counting_iterator<u32>(num_elem),
		// 									d_mat_batch,
		// 									thrust::placeholders::_1 * 10 + i * 10 + elem_metric);
		GetElemInvJ3DLoadMatrixBatchKernel<I, T><<<num_block, block_size, 0, stream[0]>>>(num_elem, d_mat_batch, elem_metric + i * 10);
		// IncrementByN<I, T*, I><<<num_block, block_size, 0, stream[0]>>>(MAX_BATCH_SIZE, d_mat_batch, MAX_BATCH_SIZE * 10);
		// thrust::async::transform(thrust::cuda::par.on(stream[0]),
		// 												thrust::make_counting_iterator<u32>(0),
		// 												thrust::make_counting_iterator<u32>(num_elem),
		// 												d_mat_batch,
		// 												GetMatBatch<I, T>(elem_metric + i * 10));


		/* Invert the Jacobian matrix */
		cublasDgetrfBatched(handle, 3, d_mat_batch, 3, pivot, info, num_elem);
		GetElemDetJKernel<<<num_block, block_size, 0, stream[0]>>>(num_elem, elem_metric + i * 10);
		cublasDgetriBatched(handle, 3, d_mat_batch, 3, pivot, d_matinv_batch, 3, info, num_elem);
		/* Copy the inverse of the Jacobian matrix to the elem_metric */
		/* elem_metric[0:9, :] = d_elem_buff[0:9, :] */
		cublasDgeam(handle,
								CUBLAS_OP_N, CUBLAS_OP_N,
								9, num_elem,
								&one,
								d_elem_buff, 10,
								&zero,
								elem_metric + i * 10, 10,
								elem_metric + i * 10, 10);
	}
	cudaStreamSynchronize(stream[0]);

	cudaStreamDestroy(stream[0]);

	cublasDestroy(handle);

	// CdamFreeDevice(info, MAX_BATCH_SIZE * sizeof(int));
	// CdamFreeDevice(d_mat_batch, MAX_BATCH_SIZE * sizeof(T*) * 2);
	// CdamFreeDevice(d_elem_buff, MAX_BATCH_SIZE * 10 * sizeof(T));
}


template <typename I, typename T, I NSHL=4, I BS=1>
struct LoadBatchFunctor {
	const I* batch_ptr, *ien;
	const T* src;
	T* dst;

	LoadBatchFunctor(const I* batch_ptr, const I* ien, const T* src, T* dst) : batch_ptr(batch_ptr), ien(ien), src(src), dst(dst) {}

	__host__ __device__ void operator()(I index) const {
		I iel = batch_ptr[index];
		#pragma unroll
		for(I i = 0; i < NSHL; ++i) {
			I node_id = ien[iel * NSHL + i];
			#pragma unroll
			for(I j = 0; j < BS; ++j) {
				dst[(index * NSHL + i)	* BS + j] = src[node_id * BS + j];
			}
		}
	}
	
};

template <typename I, typename T, I NSHL=4, I BS=1>
struct SaveBatchFunctor {
	const I* batch_ptr, *ien;
	const T* src;
	T* dst;

	SaveBatchFunctor(const I* batch_ptr, const I* ien, const T* src, T* dst) : batch_ptr(batch_ptr), ien(ien), src(src), dst(dst) {}

	__host__ __device__ void operator()(I index) const {
		I iel = batch_ptr[index];
		#pragma unroll
		for(I i = 0; i < NSHL; ++i) {
			I node_id = ien[iel * NSHL + i];
			#pragma unroll
			for(I j = 0; j < BS; ++j) {
				dst[node_id * BS + j] += src[(index * NSHL + i) * BS + j];
			}
		}
	}
};

__BEGIN_DECLS__



/**
* @brief Load the values from the source array to the destination array
* @param[in]  count The number of values to be loaded
* @param[in]  index The index of the values to be loaded, whose size is (count*inc,)
* @param[in]  inc   The increment of the index
* @param[in]  bs    The length of the block
* @param[in]  src   The source array to be loaded. The array is column-majored of size (block_length, *)
* @param[out] dst   The destination array to be loaded. The array is column-majored of size (block_length, count)
*/
static __global__ void
LoadValueKernel(u32 count, const u32* index, u32 inc, u32 block_length, const f64* src, f64* dst) {

	i32 idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if(idx >= count) return;

	const u32* index_ptr = index + idx * inc;
	f64* dst_ptr = dst + idx * block_length;
	while(block_length >= 4) {
		*((double4*)dst_ptr) = *((double4*)(src + index_ptr[0]));
		block_length -= 4;
		dst_ptr += 4;
		index_ptr += 4;
	}
	if(block_length >= 2) {
		*((double2*)dst_ptr) = *((double2*)(src + index_ptr[0]));
		block_length -= 2;
		dst_ptr += 2;
		index_ptr += 2;
	}
	if(block_length >= 1) {
		*dst_ptr = src[index_ptr[0]];
	}

}

/**
 * @brief Load the values from the source array and apply axpy operation to the destination array
 */
static __global__ void
LoadValueAxpyKernel(u32 count, const u32* index, u32 inc, u32 block_length, const f64* src, f64* dst, f64 alpha) {
	i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= count) return;

	const u32* index_ptr = index + idx * inc;
	f64* dst_ptr = dst + idx * block_length;
	while(block_length >= 4) {
		dst_ptr[0] += alpha * src[index_ptr[0]];
		dst_ptr[1] += alpha * src[index_ptr[1]];
		dst_ptr[2] += alpha * src[index_ptr[2]];
		dst_ptr[3] += alpha * src[index_ptr[3]];
		block_length -= 4;
		dst_ptr += 4;
		index_ptr += 4;
	}
	if(block_length >= 2) {
		dst_ptr[0] += alpha * src[index_ptr[0]];
		dst_ptr[1] += alpha * src[index_ptr[1]];
		block_length -= 2;
		dst_ptr += 2;
		index_ptr += 2;
	}
	if(block_length >= 1) {
		*dst_ptr += alpha * src[index_ptr[0]];
	}
}

static __global__ void
GetShapeGradKernel(u32 num_elem, const f64* elem_invJ, f64* shgradl) {
	i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num_elem) return;

	const f64* elem_invJ_ptr = elem_invJ + idx * 10;
	f64* shgradl_ptr = shgradl + idx * 12;
	f64 dxidx[9] = {elem_invJ_ptr[0], elem_invJ_ptr[1], elem_invJ_ptr[2],
									elem_invJ_ptr[3], elem_invJ_ptr[4], elem_invJ_ptr[5],
									elem_invJ_ptr[6], elem_invJ_ptr[7], elem_invJ_ptr[8]};

	#pragma unroll
	for(u32 s = 0; s < 4; ++s) {
		f64 shgradu[3] = {shlgradu[s*3+0], shlgradu[s*3+1], shlgradu[s*3+2]};
		shgradl_ptr[s*3+0] = shgradu[0] * dxidx[0] + shgradu[1] * dxidx[3] + shgradu[2] * dxidx[6];
		shgradl_ptr[s*3+1] = shgradu[0] * dxidx[1] + shgradu[1] * dxidx[4] + shgradu[2] * dxidx[7];
		shgradl_ptr[s*3+2] = shgradu[0] * dxidx[2] + shgradu[1] * dxidx[5] + shgradu[2] * dxidx[8];
	}
}


/**
 * @brief Assemble RHS for the tetrahedral elements
 * @param[in] ne The number of elements
 * @param[in] elem_metric The metric of the elements. The metric is column-majored of size (10, ne), where the first 9 elements are the inverse of the Jacobian matrix and the last element is the determinant of the Jacobian matrix
 * @param[in] shgradg The gradient of the shape functions. The gradient is column-majored of size (12, ne)
 * @param[in] qr_dwgalpha The quadrature weights of the gradient of the field. The array is column-majored of size (NQR, ne)
 * @param[in] qr_wgalpha The quadrature weights of the field. The array is column-majored of size (NQR, ne)
 * @param[in] qr_wggradalpha The quadrature weights of the gradient of the field. The array is column-majored of size (3, ne)
 * @param[out] elem_F The elementwise residual vector. The array is column-majored of size (NSHL, ne)
 */
void IntElemAssVec(u32 ne,
									 const f64* elem_metric, const f64* shgradg,
									 const f64* qr_dwgalpha, const f64* qr_wgalpha, const f64* qr_wggradalpha,
									 f64* elem_F) {
	/* 0. elem_F[0:NSHL, :] = 0.0 */
	cudaMemset(elem_F, 0, ne * sizeof(f64) * 4);
	/* 1. elem_F[0:NSHL, :] += \sum_{q=0}^{NQR-1} qr_dwgalpha[q, :] * gw[q] * detJ * shlu[q, 0:NSHL] */
	cublasHandle_t handle;
	cublasCreate(&handle);
	const f64 one = 1.0, zero = 0.0;
	for(u32 q = 0; q < NQR; ++q) {
		cublasDger(handle, 4, ne, &one, qr_dwgalpha + q, NQR, shgradg + q * 4, 1, elem_F, 4);
	}
	

	cublasDestroy(handle);
}


void AssembleSystemTet(Mesh3D *mesh,
											 Field *wgold, Field* dwgold, Field* dwg,
											 f64* F, Matrix* J) {

	const i32 NSHL = 4;
	const i32 MAX_BATCH_SIZE = 1 << 10;
	const u32 num_tet = Mesh3DNumTet(mesh);
	const u32 num_node = Mesh3DNumNode(mesh);
	const Mesh3DData* dev = Mesh3DDevice(mesh);
	const u32* ien = Mesh3DDataTet(dev);
	const f64* xg = Mesh3DDataCoord(dev);

	f64* wgold_dptr = ArrayData(FieldDevice(wgold));
	f64* dwgold_dptr = ArrayData(FieldDevice(dwgold));
	f64* dwg_dptr = ArrayData(FieldDevice(dwg));



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

	f64 alpha_m = kALPHAM, one_minus_alpha_m = 1.0 - kALPHAM;
	f64* dwgalpha_dptr = (f64*)CdamMallocDevice(num_node * sizeof(f64));
	/* dwgalpha_ptr[:] = (1-alpha_m) * dwgold[:] + alpha_m * dwg[:] */
	cudaMemset(dwgalpha_dptr, 0, num_node * sizeof(f64));
	cublasDaxpy(handle, num_node, &one_minus_alpha_m, dwgold_dptr, 1, dwgalpha_dptr, 1);
	cublasDaxpy(handle, num_node, &alpha_m, dwg_dptr, 1, dwgalpha_dptr, 1);


	f64 fact1 = kDT * kALPHAF * kGAMMA, fact2 = kDT * kALPHAF * (1.0 - kGAMMA); 
	f64* wgalpha_dptr = (f64*)CdamMallocDevice(num_node * sizeof(f64));
	/* wgalpha_ptr[:] = wgold[:] + kDT * kALPHAF * (1.0 - kGAMMA) * dwgold[:] + kDT * kALPHAF * kGAMMA * dwg[:] */
	cublasDcopy(handle, num_node, wgold_dptr, 1, wgalpha_dptr, 1);
	cublasDaxpy(handle, num_node, &fact1, dwg_dptr, 1, wgalpha_dptr, 1);
	cublasDaxpy(handle, num_node, &fact2, dwgold_dptr, 1, wgalpha_dptr, 1);

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

	f64* buffer = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NSHL);
	f64* elem_invJ = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * 10);
	f64* shgradg = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NSHL * 3);

	f64* qr_wgalpha = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NQR);
	f64* qr_dwgalpha = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NQR);
	f64* qr_wggradalpha = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * 3);

	i32 num_thread = 256;
	i32 num_block = (max_batch_size + num_thread - 1) / num_thread;

	// f64* elem_buff = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NSHL * NSHL);
	f64* elem_F = NULL;
	f64* elem_J = NULL;

	if(F) {
		elem_F = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NSHL);
	}
	if(J) {
		elem_J = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NSHL * NSHL);
	}

	f64 time_len[6] = {0, 0, 0, 0, 0, 0};
	std::chrono::steady_clock::time_point start, end;

	/* Assume all 2D arraies are column-majored */
	for(u32 b = 0; b < mesh->num_batch; ++b) {

		start = std::chrono::steady_clock::now();
		// batch_size = CountValueColorLegacy(mesh->color, num_tet, b);
		batch_size = mesh->batch_offset[b+1] - mesh->batch_offset[b];
		// batch_size = CountValueColor(mesh->color, num_tet, b, count_value_buffer);

		if(batch_size == 0) {
			break;
		}


		// FindValueColor(mesh->color, num_tet, b, batch_index_ptr);
		// cudaMemcpy(batch_index_ptr, mesh->batch_ind + mesh->batch_offset[b], batch_size * sizeof(u32), cudaMemcpyDeviceToDevice);
		batch_index_ptr = mesh->batch_ind + mesh->batch_offset[b];
		end = std::chrono::steady_clock::now();
		// CUGUARD(cudaGetLastError());
		time_len[0] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		/* 0. Calculate the element metrics */
		/* 0.0. Get dxi/dx and det(dxi/dx) */

		start = std::chrono::steady_clock::now();
		GetElemInvJ3D(batch_size, ien, batch_index_ptr, xg, elem_invJ);
		/* 0.1. Calculate the gradient of the shape functions */
		/* shgradg[0:3, 0:NSHL, e] = elem_invJ[0:3, 0:3, e].T @ d_shlgradu[0:3, 0:NSHL, e] for e in range(batch_size) */
		cublasDgemmStridedBatched(handle,
															CUBLAS_OP_T, CUBLAS_OP_N,
															3, NSHL, 3,
															&one,
															elem_invJ, 3, 10ll,
															d_shlgradu, 3, 0ll,
															&zero,
															shgradg, 3, (long long)(NSHL * 3),
															batch_size);
		// CUGUARD(cudaGetLastError());
		/* 1. Interpolate the field values */  
		// cudaMemset(qr_wgalpha, 0, max_batch_size * sizeof(f64));
		// cudaMemset(qr_wggradalpha, 0, max_batch_size * sizeof(f64) * 3);
		// CUGUARD(cudaGetLastError());
		end = std::chrono::steady_clock::now();
		time_len[1] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	
		start = std::chrono::steady_clock::now();
	
		/* 1.0. Calculate the field value on the vertices */
		LoadElementValueKernel<u32, f64><<<(batch_size + num_thread - 1) / num_thread, num_thread>>>(batch_size,
																																																 ien, batch_index_ptr,
																																																 wgalpha_dptr, buffer);
		// thrust::for_each(thrust::device,
		// 								 thrust::make_counting_iterator<u32>(0),
		// 								 thrust::make_counting_iterator<u32>(batch_size),
		// 								 LoadBatchFunctor<u32, f64, NSHL, 1>(batch_index_ptr, ien, wgalpha_dptr, buffer));
											
		// CUGUARD(cudaGetLastError());
		/* 1.1. Calculate the gradient*/
		/* qr_wggradalpha[0:3, e] = shgradg[0:3, 0:NSHL, e] @ buffer[0:NSHL, e] for e in range(batch_size) */
		cublasDgemvStridedBatched(handle, CUBLAS_OP_N,
															3, NSHL,
															&one,
															shgradg, 3, (long long)(NSHL * 3),
															buffer, 1, (long long)NSHL,
															&zero,
															qr_wggradalpha, 1, 3ll,
															batch_size);
		// CUGUARD(cudaGetLastError());

		/* 1.2. Calculate the field value on the quadrature */
		/* qr_wgalpha[0:NQR, 0:batch_size] = d_shlu[0:NQR, 0:NSHL] @ buffer[0:NSHL, 0:batch_size] */
		cublasDgemm(handle,
								CUBLAS_OP_N, CUBLAS_OP_N,
								NQR, batch_size, NSHL,
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
		LoadElementValueKernel<u32, f64><<<(batch_size + num_thread - 1) / num_thread, num_thread>>>(batch_size,
																																																 ien, batch_index_ptr,
																																																 dwgalpha_dptr, buffer);
		// thrust::for_each(thrust::device,
		// 								 thrust::make_counting_iterator<u32>(0),
		// 								 thrust::make_counting_iterator<u32>(batch_size),
		// 								 LoadBatchFunctor<u32, f64, NSHL, 1>(batch_index_ptr, ien, dwgalpha_dptr, buffer));
		// CUGUARD(cudaGetLastError());
		/* qr_dwgalpha[0:NQR, 0:batch_size] = d_shlu[0:NQR, 0:NSHL] @ buffer[0:NSHL, 0:batch_size] */
		cublasDgemm(handle,
								CUBLAS_OP_N, CUBLAS_OP_N,
								NQR, batch_size, NSHL,
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
		AssemleWeakFormKernel<u32, f64, NSHL><<<(batch_size * NSHL + num_thread - 1) / num_thread, num_thread>>>(batch_size,
																																																						 elem_invJ,
																																																						 qr_wgalpha, qr_dwgalpha, qr_wggradalpha,
																																																						 shgradg,
																																																						 elem_F, elem_J);
		end = std::chrono::steady_clock::now();
		time_len[4] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
		/* 3. Assemble the global residual vector */
		start = std::chrono::steady_clock::now();
		if(F) {
			ElemRHSLocal2GlobalKernel<u32, f64><<<(batch_size + num_thread - 1) / num_thread, num_thread>>>(batch_size,
																																																			batch_index_ptr, ien,
																																																			elem_F, F);
			// thrust::for_each(thrust::device,
			// 								 thrust::make_counting_iterator<u32>(0),
			// 								 thrust::make_counting_iterator<u32>(batch_size),
			// 								 SaveBatchFunctor<u32, f64, NSHL, 1>(batch_index_ptr, ien, elem_F, F));
		}
		/* 4. Assemble the global residual matrix */
		if(J) {
			MatrixAddElementLHS(J, NSHL, 1, batch_size, ien, batch_index_ptr, elem_J);
		}
		end = std::chrono::steady_clock::now();
		time_len[5] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	}
	printf("Time[0]: GenerateBatch: %10.4f ms\n", time_len[0]* 1e-6);
	printf("Time[1]: GetElemInvJ3D: %10.4f ms\n", time_len[1]* 1e-6);
	printf("Time[2]: Interpolate wg: %10.4f ms\n", time_len[2]* 1e-6);
	printf("Time[3]: Interpolate dwg: %10.4f ms\n", time_len[3]* 1e-6);
	printf("Time[4]: AssembleWeakForm: %10.4f ms\n", time_len[4]* 1e-6);
	printf("Time[5]: AssembleGlobal: %10.4f ms\n", time_len[5]* 1e-6);

	CdamFreeDevice(buffer, max_batch_size * sizeof(f64));
	CdamFreeDevice(elem_invJ, max_batch_size * sizeof(f64) * 10);
	CdamFreeDevice(shgradg, max_batch_size * sizeof(f64) * NSHL * 3);


	CdamFreeDevice(elem_F, max_batch_size * sizeof(f64) * NSHL);
	CdamFreeDevice(elem_J, max_batch_size * sizeof(f64) * NSHL * NSHL);
	// CdamFreeDevice(elem_buff, max_batch_size * sizeof(f64) * NSHL * NSHL);

	CdamFreeDevice(qr_wgalpha, num_tet * sizeof(f64));
	CdamFreeDevice(qr_dwgalpha, num_tet * sizeof(f64));
	CdamFreeDevice(qr_wggradalpha, num_tet * sizeof(f64) * 3);
	// CdamFreeDevice(batch_index_ptr, max_batch_size * sizeof(u32));
	cublasDestroy(handle);
}

__END_DECLS__
