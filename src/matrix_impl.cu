#include "alloc.h"
#include "matrix.h"


template <typename ValueType, typename IndexType>
__global__ void
MatrixCSRGetDiagKernel(const ValueType* val, const IndexType* row_ptr, const IndexType* col_idx, ValueType* diag, const IndexType num_row) {
	const IndexType i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < num_row) {
		for(IndexType j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
			if(col_idx[j] == i) {
				diag[i] = val[j];
				break;
			}
		}
	}
}


template <typename IndexType, typename ValueType>
__global__ void
MatrixCSRAddElementLHSKernel(ValueType* matval,
															const IndexType* row_ptr, const IndexType* col_ind, IndexType num_row, IndexType num_col,
															IndexType NSHL, IndexType BS, IndexType num_batch, const IndexType* ien, const IndexType* batch_ptr,
															const ValueType* val) {
	const IndexType idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= num_batch) return;
	IndexType iel = batch_ptr[idx];
	val += NSHL*NSHL*BS*BS*iel;

	IndexType d_ien[8];
	for(IndexType aa = 0; aa < NSHL; ++aa) {
		d_ien[aa] = ien[iel * NSHL + aa];
	}
	
	for(IndexType aa = 0; aa < NSHL; ++aa) {
		for(IndexType ii = 0; ii < BS; ++ii) {
			IndexType ir = d_ien[aa] * BS + ii;
			IndexType row_start = row_ptr[ir];
			IndexType row_end = row_ptr[ir + 1];
			for(IndexType bb = 0; bb < NSHL; ++bb) {
				for(IndexType jj = 0; jj < BS; ++jj) {
					IndexType ic = d_ien[bb] * BS + jj;
					for(IndexType j = row_start; j < row_end; j++) {
						if(col_ind[j] == ic) {
							matval[j] += val[(aa * NSHL + ii) * BS * NSHL + bb * BS + jj];
							break;
						}
					}
				}
			}
		}
	}
	
}



__BEGIN_DECLS__


void MatrixCSRGetDiagGPU(const value_type* val, const index_type* row_ptr, const index_type* col_idx, value_type* diag, const index_type num_row) {
	const size_t block_dim = 256;
	const size_t grid_dim = (num_row + block_dim - 1) / block_dim;
	MatrixCSRGetDiagKernel<value_type, index_type><<<grid_dim, block_dim>>>(val, row_ptr, col_idx, diag, num_row);
}


void MatrixCSRAddElementLHSGPU(value_type* matval,
															 const index_type* row_ptr, const index_type* col_ind, index_type num_row, index_type num_col,
															 index_type nshl, index_type bs, index_type num_batch, const index_type* ien, const index_type* batch_ptr,
															 const value_type* val) {
	const size_t block_dim = 256 * 4;
	const size_t grid_dim = (num_batch + block_dim - 1) / block_dim;

	MatrixCSRAddElementLHSKernel<index_type, value_type><<<grid_dim, block_dim>>>(matval, row_ptr, col_ind, num_row, num_col,
																																								nshl, bs, num_batch, ien, batch_ptr, val);
}


__END_DECLS__
