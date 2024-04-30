#include "alloc.h"
#include "matrix.h"


template<typename value_type, typename index_type>
__global__ void
MatrixCSRGetDiagKernel(const value_type* val, const index_type* row_ptr, const index_type* col_idx, value_type* diag, const index_type num_row) {
	const index_type i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < num_row) {
		for(index_type j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
			if(col_idx[j] == i) {
				diag[i] = val[j];
				break;
			}
		}
	}
}

__BEGIN_DECLS__


void MatrixCSRGetDiagGPU(const value_type* val, const index_type* row_ptr, const index_type* col_idx, value_type* diag, const index_type num_row) {
	const size_t block_size = 256;
	const size_t num_block = (num_row + block_size - 1) / block_size;
	MatrixCSRGetDiagKernel<value_type, index_type><<<num_block, block_size>>>(val, row_ptr, col_idx, diag, num_row);
}

__END_DECLS__
