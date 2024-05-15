#include <type_traits>
#include <cublas_v2.h>

#include "alloc.h"
#include "indexing.h"
#include "Mesh.h"
#include "Field.h"
#include "Array.h"
#include "csr.h"

#include "assemble.h"

#define kRHOC (0.5)
#define kDT (0.1)
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


template<typename I, typename T, I bs=1, I NSHL=4>
__global__ void LoadElementValueKernel(I n, const I* ien, const I* index, const T* src, T* dst) {
	I idx = blockIdx.x * blockDim.x + threadIdx.x;
	I num_thread = blockDim.x * gridDim.x;
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


template <typename I, typename T, I bs=1, I NSHL=4> __global__ void
ElemRHSLocal2GlobalKernel(I num_elem, const I* iel, const I* ien, const T* elem_F, T* F) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for(int j = 0; j < num_elem * NSHL; j += blockDim.x * gridDim.x) {
		int vertex_id = ien[iel[j] * NSHL + j % NSHL];
		#pragma unroll
		for(int j = 0; j < bs; ++j) {
			F[vertex_id * bs + j] += elem_F[j * bs + j];
		}
	}
}


template<typename I, typename T, I NSHL=4>
__global__ void GetElemJ3DKernel(I ne, const I* ien, const I* index, const T* xg, T* elemJ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	I elem_id = index[idx];
	const T* xg_ptr0 = xg + ien[elem_id*NSHL+0] * 3;
	const T* xg_ptr1 = xg + ien[elem_id*NSHL+1] * 3;
	const T* xg_ptr2 = xg + ien[elem_id*NSHL+2] * 3;
	const T* xg_ptr3 = xg + ien[elem_id*NSHL+3] * 3;
	
	T* elemJ_ptr = elemJ + idx * 9;
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
	T* elem_detJ = elem_lu + 9; /* elem_detJ is the determinant of the Jacobian matrix */
	*elem_detJ = elem_lu[0] * elem_lu[4] * elem_lu[8];
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

/**
 * @brief Calculate the inverse of the Jacobian matrix of the element
 * @param[in] ne   The number of elements
 * @param[in] ien  The index of the element in the mesh
 * @param[in] xg   The global coordinates of the nodes
 * @param[out] elem_metric The inverse of the Jacobian matrix of the element
 */
template<typename I, typename T, I NSHL=4>
void GetElemInvJ3DKernel(I ne, const I* ien, const I* index, const T* xg, T* elem_metric) {
	int block_size = 1024;
	int num_block = (ne + block_size - 1) / block_size;
	cublasHandle_t handle;
	cublasCreate(&handle);
	GetElemJ3DKernel<<<num_block, block_size>>>(ne, ien, index, xg, elem_metric);

	const int BATCH_SIZE = 128;
	int* info = (int*)CdamMallocDevice(BATCH_SIZE * sizeof(int));
	int* pivot = (int*)CdamMallocDevice(BATCH_SIZE * 3 * sizeof(int));

	T* elem_metric_batch[BATCH_SIZE];
	T* elem_batch[BATCH_SIZE];

	T** d_elem_metric_batch = (T**)CdamMallocDevice(BATCH_SIZE * sizeof(T*));
	T** d_elem_batch = (T**)CdamMallocDevice(BATCH_SIZE * sizeof(T*));


	T* elem_batch_buff = (T*)CdamMallocDevice(BATCH_SIZE * 10 * sizeof(T));
	for(int j = 0; j < BATCH_SIZE; ++j) {
		elem_batch[j] = elem_batch_buff + j * 10;
	}
	cudaMemcpy(d_elem_batch, elem_batch, BATCH_SIZE * sizeof(T*), cudaMemcpyHostToDevice);


	for(int i = 0; i < ne; i += BATCH_SIZE) {
		int num_elem = (i + BATCH_SIZE < ne) ? BATCH_SIZE : ne - i;
		for(int j = 0; j < num_elem; ++j) {
			elem_metric_batch[j] = elem_metric+ (i + j) * 10;
		}

		cudaMemcpy(d_elem_metric_batch, elem_metric_batch, BATCH_SIZE * sizeof(T*), cudaMemcpyHostToDevice);

		if constexpr (std::is_same_v<T, double>) {
			cublasDgetrfBatched(handle, 3, elem_metric_batch, 3, pivot, info, num_elem);
			GetElemDetJKernel<<<num_block, block_size>>>(num_elem, elem_metric + i * 10);
			cublasDgetriBatched(handle, 3, elem_metric_batch, 3, pivot, elem_batch, 3, info, num_elem);
		}
		else if constexpr (std::is_same_v<T, float>) {
		}
		else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
		}
		else if constexpr (std::is_same_v<T, cuFloatComplex>) {
		}

		CopyElemJacobiansKernel<I, T><<<num_block, block_size>>>((I)BATCH_SIZE, (const T**)elem_batch, elem_metric + i * 10);
	}

	CdamFreeDevice(info, BATCH_SIZE * sizeof(int));
	CdamFreeDevice(pivot, BATCH_SIZE * 3 * sizeof(int));
	CdamFreeDevice(elem_batch_buff, BATCH_SIZE * 10 * sizeof(T));
	CdamFreeDevice(d_elem_metric_batch, BATCH_SIZE * sizeof(T*));
	CdamFreeDevice(d_elem_batch, BATCH_SIZE * sizeof(T*));
}



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
		// *((double4*)dst_ptr) = fma(*((double4*)(src + index_ptr[0])), alpha, *((double4*)dst_ptr));
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
static __global__ void
AssemleTetWeakFormKernel(u32 num_elem, const u32* iel, const f64* elem_invJ,
												 const f64* qr_wgalpha, const f64* qr_dwgalpha, const f64* qr_wggradalpha,
												 const f64* shgradl,
												 f64* elem_F, f64* elem_J) {
	const int NSHL = 4;
	i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num_elem) return;

	const f64* elem_invJ_ptr = elem_invJ + idx * 10;
	const f64 detJ = elem_invJ_ptr[9];
	const f64* qr_wgalpha_ptr = qr_wgalpha + idx * 4;
	const f64* qr_dwgalpha_ptr = qr_dwgalpha + idx * 4;
	const f64* qr_wggradalpha_ptr = qr_wggradalpha + idx * 3;
	const f64* shgradl_ptr = shgradl + idx * 12;

	f64* elem_F_ptr = elem_F + idx * 4;
	f64* elem_J_ptr = elem_J + idx * 16;

	f64 F_ptr[4+16] = {0.0, 0.0, 0.0, 0.0,
								 0.0, 0.0, 0.0, 0.0,
								 0.0, 0.0, 0.0, 0.0,
								 0.0, 0.0, 0.0, 0.0,
								 0.0, 0.0, 0.0, 0.0};
	f64* J_ptr = F_ptr + 4;
	
	/* Weak form: */
	/* F_A[0:NSHL] += (gw[0:NQR] * detJ * qr_dwgalpha_ptr[0:NQR]) @ shl[0:NQR, 0:NSHL] */
	/* F_A[0:NSHL] += sum(gw[0:NQR]) * detJ * qr_wggradalpha_ptr[0:3] @ shgradl[0:3, 0:NSHL] */
	/* J_AB[0:NSHL, 0:NSHL] += kALPHAM * detJ * shl[0,NQR, NSHL].T @ (gw[0:NQR, None] * shl[0:NQR, 0:NSHL]) */
	/* J_AB[0:NSHL, 0:NSHL] += kDT * kALPHAF * kGAMMA * sum(gw[0:NQR]) * detJ 
													 * shgradl[0:3, 0:NSHL].T @ shgradl[0:3, 0:NSHL] */

	if(elem_F) {
		#pragma unroll
		for(u32 q = 0; q < NQR; ++q) {
			f64 qr_dwgalpha_q = qr_dwgalpha_ptr[q];
			f64 qr_wggradalpha_q = qr_wggradalpha_ptr[q];
			#pragma unroll
			for(u32 s = 0; s < NSHL; ++s) {
				F_ptr[s] += gw[q] * detJ * qr_dwgalpha_q * shgradl_ptr[q*3+s];
				F_ptr[s] += gw[q] * detJ * qr_wggradalpha_q * shgradl_ptr[q*3+s];
			}
		}

		#pragma unroll
		for(u32 s = 0; s < NSHL; ++s) {
			#pragma unroll
			for(u32 t = 0; t < NSHL; ++t) {
				elem_F_ptr[s] += F_ptr[s*NSHL+t];
			}
		}
	}
	if(elem_J) {
		#pragma unroll
		for(u32 q = 0; q < NQR; ++q) {
			#pragma unroll
			for(u32 s = 0; s < NSHL; ++s) {
				#pragma unroll
				for(u32 t = 0; t < NSHL; ++t) {
					J_ptr[s*NSHL+t] += kALPHAM * detJ * shlgradu[q*3+s] * gw[q] * shlgradu[q*3+t];
					J_ptr[s*NSHL+t] += kDT * kALPHAF * kGAMMA * detJ * shgradl_ptr[q*3+s] * shgradl_ptr[q*3+t];
				}
			}
		}

		#pragma unroll
		for(u32 s = 0; s < NSHL; ++s) {
			#pragma unroll
			for(u32 t = 0; t < NSHL; ++t) {
				elem_J_ptr[s*NSHL+t] = J_ptr[s*NSHL+t];
			}
		}
	
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
	const u32 num_tet = Mesh3DNumNode(mesh);
	const Mesh3DData* dev = Mesh3DDevice(mesh);
	const u32* ien = Mesh3DDataTet(dev);
	const f64* xg = Mesh3DDataCoord(dev);

	f64* wgold_dptr = ArrayData(FieldDevice(wgold));
	f64* dwgold_dptr = ArrayData(FieldDevice(dwgold));
	f64* dwg_dptr = ArrayData(FieldDevice(dwg));


	u32* batch_index_ptr = NULL;
	u32 max_batch_size = 0, batch_size = 0;
	cublasHandle_t handle;
	const f64 one = 1.0, zero = 0.0, minus_one = -1.0;

	cublasCreate(&handle);
	f64* A_batch[MAX_BATCH_SIZE];
	f64* B_batch[MAX_BATCH_SIZE];
	f64* C_batch[MAX_BATCH_SIZE];

	for(u32 b = 0; b < mesh->num_color; ++b) {
		batch_size = CountValueColor(mesh->color, num_tet, b);
		if (batch_size > max_batch_size) {
			max_batch_size = batch_size;
		}
	}
	batch_index_ptr = (u32*)CdamMallocDevice(max_batch_size * sizeof(u32));

	f64* buffer = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64));
	f64* elem_invJ = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * 10);
	f64* shgradg = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NSHL * 3);

	f64* elem_F = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NSHL);
	f64* elem_J = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NSHL * NSHL);

	f64* qr_wgalpha = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NQR);
	f64* qr_dwgalpha = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * NQR);
	f64* qr_wggradalpha = (f64*)CdamMallocDevice(max_batch_size * sizeof(f64) * 3);


	/* Assume all 2D arraies are column-majored */
	for(u32 b = 0; b < mesh->num_color; ++b) {

		i32 num_thread = 256;
		i32 num_block = (batch_size + num_thread - 1) / num_thread;
	
		batch_size = CountValueColor(mesh->color, num_tet, b);
		FindValueColor(mesh->color, num_tet, b, batch_index_ptr);
		/* 0. Calculate the element metrics */
		/* 0.0. Get dxi/dx and det(dxi/dx) */
		GetElemInvJ3DKernel(batch_size, ien, batch_index_ptr, xg, elem_invJ);
		/* 0.1. Calculate the gradient of the shape functions */
		for(u32 e = 0; e < batch_size; ++e) {
			A_batch[e] = elem_invJ + batch_index_ptr[e] * 10;
			B_batch[e] = shlgradu;
			C_batch[e] = shgradg + e * 12;
		}
		/* C_batch[:, :, e] = A_batch[0:3, 0:3, e].T @ B_batch[0:3, 0:NSHL, e] for e in range(new_batch_seze) */
		cublasDgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3, NSHL, 3, &one, A_batch, 3, B_batch, 3, &zero, C_batch, 3, batch_size);

		/* 1. Interpolate the field values */  
		cudaMemset(qr_wgalpha, 0, max_batch_size * sizeof(f64));
		cudaMemset(qr_dwgalpha, 0, max_batch_size * sizeof(f64));
		cudaMemset(qr_wggradalpha, 0, max_batch_size * sizeof(f64) * 3);
	
		/* 1.0. Calculate the field value on the vertices */
		LoadElementValueKernel<<<num_block, num_thread>>>(batch_size, ien, batch_index_ptr, wgold_dptr, buffer);
		LoadElementValueAXPYKernel<<<num_block, num_thread>>>(batch_size, ien, batch_index_ptr, dwgold_dptr, buffer, kDT * kALPHAF * (1.0 - kGAMMA));
		LoadElementValueAXPYKernel<<<num_block, num_thread>>>(batch_size, ien, batch_index_ptr, dwg_dptr, buffer, kDT * kALPHAF * kGAMMA);

		/* 1.1. Calculate the gradient*/
		for(u32 e = 0; e < batch_size; ++e) {
			A_batch[e] = shgradg + e * 12;
			B_batch[e] = buffer + e * NSHL;
			C_batch[e] = qr_wggradalpha + e * 3;
		}
		/* C_batch[0:3, e] = A_batch[0:3, 0:NSHL, i] @ B_batch[0:NSHL, e] for e in range(batch_size) */	
		cublasDgemvBatched(handle, CUBLAS_OP_N, 3, NSHL, &one, A_batch, 3, B_batch, 1, &zero, C_batch, 1, batch_size);

		/* 1.2. Calculate the field value on the quadrature */
		/* qr_wgalpha[0:NQR, :] = shlu[0:NQR, 0:NSHL] @ buffer[0:NSHL, :] */
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NQR, batch_size, NSHL, &one, shlu, NQR, buffer, NSHL, &zero, qr_wgalpha, NQR);

		/* 1.3. Calculate the field value of qr_dwgalpha */
		cudaMemset(qr_dwgalpha, 0, num_tet * sizeof(f64) * NQR);
		LoadElementValueAXPYKernel<<<num_block, num_thread>>>(batch_size, ien, batch_index_ptr, dwgold_dptr, buffer, 1.0 - kALPHAM);
		LoadElementValueAXPYKernel<<<num_block, num_thread>>>(batch_size, ien, batch_index_ptr, dwg_dptr, buffer, kALPHAM);
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NQR, batch_size, NSHL, &one, shlu, NQR, buffer, NSHL, &zero, qr_dwgalpha, NQR);


		/* 2. Calculate the elementwise residual and jacobian */
		AssemleTetWeakFormKernel<<<num_block, num_thread>>>(batch_size, batch_index_ptr, elem_invJ, qr_wgalpha, qr_dwgalpha, qr_wggradalpha, shgradg, elem_F, elem_J);
		/* 3. Assemble the global residual vector */
		if(F) {
			ElemRHSLocal2GlobalKernel<<<num_block, num_thread>>>(batch_size, batch_index_ptr, ien, elem_F, F);
		}
		/* 4. Assemble the global residual matrix */
		if(J) {
		}
	}
	CdamFreeDevice(buffer, num_tet * sizeof(f64));
	CdamFreeDevice(elem_invJ, num_tet * sizeof(f64) * 10);
	CdamFreeDevice(shgradg, num_tet * sizeof(f64) * NSHL * 3);

	CdamFreeDevice(elem_F, num_tet * sizeof(f64) * NSHL);
	CdamFreeDevice(elem_J, num_tet * sizeof(f64) * NSHL * NSHL);

	CdamFreeDevice(qr_wgalpha, num_tet * sizeof(f64));
	CdamFreeDevice(qr_dwgalpha, num_tet * sizeof(f64));
	CdamFreeDevice(qr_wggradalpha, num_tet * sizeof(f64) * 3);
	CdamFreeDevice(batch_index_ptr, batch_size * sizeof(u32));
	cublasDestroy(handle);
}

__END_DECLS__
