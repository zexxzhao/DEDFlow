
#include "common.h"

#include "pc.cuh"

__BEGIN_DECLS__

/*
 * y = x / diag(A)
 */
__global__ __device__ void
PCJacobiApplyKernel(u32 n, u32 nnz, f64* data, u32* row_ptr, u32* col_idx, f64* x, f64* y) {
	const u32 size = blockDim.x * gridDim.x;
	u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	for(u32 i = idx; i < n; i += size) {
		if (i >= n) {
			break;
		}
		for(u32 j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
			if (col_idx[j] == i) {
				y[i] = x[i] / data[j];
				break;
			}
		}
	}
}

__global__ __device__ void
PCJacobiApplyInPlaceKernel(u32 n, u32 nnz, f64* data, u32* row_ptr, u32* col_idx, f64* x) {
	const u32 size = blockDim.x * gridDim.x;
	u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	for(u32 i = idx; i < n; i += size) {
		if (i >= n) {
			break;
		}
		for(u32 j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
			if (col_idx[j] == i) {
				x[i] /= data[j];
				break;
			}
		}
	}
}

__END_DECLS__
