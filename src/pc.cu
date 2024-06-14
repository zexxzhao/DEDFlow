
#include "common.h"

#include "pc.h"

__BEGIN_DECLS__

static __global__ void
PCJacobiApplyKernel(index_type n, index_type nnz, f64* data, index_type* row_ptr, index_type* col_idx, f64* x, f64* y) {
	const index_type size = blockDim.x * gridDim.x;
	index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
	for(index_type i = idx; i < n; i += size) {
		if (i >= n) {
			break;
		}
		for(index_type j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
			if (col_idx[j] == i) {
				y[i] = x[i] / data[j];
				break;
			}
		}
	}
}

static __global__ void
PCJacobiApplyInPlaceKernel(index_type n, index_type nnz, f64* data, index_type* row_ptr, index_type* col_idx, f64* x) {
	const index_type size = blockDim.x * gridDim.x;
	index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
	for(index_type i = idx; i < n; i += size) {
		if (i >= n) {
			break;
		}
		for(index_type j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
			if (col_idx[j] == i) {
				x[i] /= data[j];
				break;
			}
		}
	}
}

void PCJacobiDevice(index_type n, index_type nnz, f64* data, index_type* row_ptr, index_type* col_idx, f64* x, f64* y) {
	i32 block_size = 1024;
	i32 num_blocks = (n + block_size - 1) / block_size;
	ASSERT(x != y && "x and y must be different");
	PCJacobiApplyKernel<<<num_blocks, block_size>>>(n, nnz, data, row_ptr, col_idx, x, y);
}

void PCJacobiInplaceDevice(index_type n, index_type nnz, f64* data, index_type* row_ptr, index_type* col_idx, f64* x) {
	i32 block_size = 1024;
	i32 num_blocks = (n + block_size - 1) / block_size;
	PCJacobiApplyInPlaceKernel<<<num_blocks, block_size>>>(n, nnz, data, row_ptr, col_idx, x);
}

__END_DECLS__
