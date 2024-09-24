#ifndef __SEQUENTIAL_MATRIX_DEVICE_H__
#define __SEQUENTIAL_MATRIX_DEVICE_H__

#include "common.h"
#include "matrix_util.h"

#ifdef CDAM_USE_CUDA
#include <cuda_runtime.h>

__BEGIN_DECLS__
void SeqMatZeroRowDenseGPU(MatOrder order,
													 value_type* data, index_type nrow, index_type ncol,
													 index_type nr, index_type* row,
													 index_type rshift, index_type cshift,
													 value_type diag, cudaStream_t stream);

void SeqMatGetSubmatDenseGPU(MatOrder order, value_type* data, index_type nrow, index_type ncol,
														 index_type nr, index_type nc, index_type* row, index_type* col,
														 index_type rshift, index_type cshift, value_type* submat, cudaStream_t stream);

void SeqMatGetDiagDenseGPU(value_type* data, index_type n,
													 value_type* diag, index_type bs, cudaStream_t stream);

void SeqMatAddElemValueBatchedDenseGPU(value_type* data, index_type nrow, index_type ncol,
																			 MatOrder order,
																			 MatStorageMethod rmap_storage, MatStorageMethod cmap_storage,
																			 index_type batch_size, index_type* batch_index_ptr,
																			 index_type* ien, index_type nshl,
																			 index_type block_row, index_type block_col,
																			 value_type* value, index_type ldv, index_type stride,
																			 cudaStream_t stream);

void SeqMatCopySubmatValueCSRGPU(value_type* src, CSRAttr* src_spy,
																 index_type nr, index_type* row, index_type nc, index_type* col,
																 value_type* dst, CSRAttr* dst_spy, cudaStream_t stream);

void SeqMatAddElemValueCSRGPU(value_type alpha, value_type* matval,
															CSRAttr* spy, MatStorageMethod rmap_storage, MatStorageMethod cmap_storage,
															index_type batch_size, index_type* batch_index_ptr,
															index_type* ien, index_type nshl,
															index_type block_row, index_type block_col,
															value_type beta, value_type* val, int lda, int stride, cudaStream_t stream);

__END_DECLS__

#endif /* CDAM_USE_CUDA */
#endif /* __SEQUENTIAL_MATRIX_DEVICE_H__ */
