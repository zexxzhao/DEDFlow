
#include "common.h"

#include "pc.cuh"

__BEGIN_DECLS__

static __global__ void
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

static __global__ void
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

void PCJacobiDevice(u32 n, u32 nnz, f64* data, u32* row_ptr, u32* col_idx, f64* x, f64* y) {
	i32 block_size = 1024;
	i32 num_blocks = (n + block_size - 1) / block_size;
	ASSERT(x != y && "x and y must be different");
	PCJacobiApplyKernel<<<num_blocks, block_size>>>(n, nnz, data, row_ptr, col_idx, x, y);
}

void PCJacobiInplaceDevice(u32 n, u32 nnz, f64* data, u32* row_ptr, u32* col_idx, f64* x) {
	i32 block_size = 1024;
	i32 num_blocks = (n + block_size - 1) / block_size;
	PCJacobiApplyInPlaceKernel<<<num_blocks, block_size>>>(n, nnz, data, row_ptr, col_idx, x);
}

__END_DECLS__
