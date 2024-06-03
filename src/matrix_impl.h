#ifndef __MATRIX_IMPL_H__
#define __MATRIX_IMPL_H__

#include "matrix.h"

__BEGIN_DECLS__

void MatrixCSRZeroRowGPU(value_type* matval,
												 index_type num_row, index_type num_col, const index_type* row_ptr, const index_type* col_ind,
												 index_type n, const index_type* row, index_type shift, value_type diag);

void MatrixCSRGetDiagGPU(const value_type* val, const index_type* row_ptr, const index_type* col_ind, value_type* diag, index_type num_row); 

void MatrixCSRSetValuesCOOGPU(value_type* matval, value_type alpha,
															index_type num_row, index_type num_col,
															const index_type* row_ptr, const index_type* col_ind,
															index_type n, const index_type* row, const index_type* col,
															const value_type* val, value_type beta);

void MatrixCSRSetValuesIndGPU(value_type* matval, value_type alpha,
															index_type n, const index_type* ind,
															const value_type* val, value_type beta);


void MatrixCSRAddElemValueBatchedGPU(value_type* matval, value_type alpha,
																		 index_type batch_size, const index_type* batch_index_ptr, const index_type* ien, index_type nshl,
																		 index_type num_row, index_type num_col, const index_type* row_ptr, const index_type* col_ind,
																		 const value_type* val, value_type beta, const index_type* mask);

void MatrixCSRAddElemValueBlockedBatchedGPU(value_type* matval, value_type alpha,
																						index_type batch_size, const index_type* batch_index_ptr, const index_type* ien, index_type nshl,
																						index_type num_row, index_type num_col, const index_type* row_ptr, const index_type* col_ind,
																						index_type block_row, index_type block_col,
																						const value_type* val, int lda, int stride, value_type beta, const index_type* mask);

void MatrixCSRAddElementLHSGPU(value_type* matval, index_type nshl, index_type bs,
															 index_type num_row, const index_type* row_ptr,
															 index_type num_col, const index_type* col_ind,
															 index_type batch_size, const index_type* batch_ptr, const index_type* ien,
															 const value_type* val, int lda);


void MatrixCSRSetValueBatchedGPU(value_type* matval, value_type alpha,
																 index_type csr_num_row, index_type csr_num_col,
																 const index_type* csr_row_ptr, const index_type* csr_col_ind,
																 index_type batch_size,
																 const index_type* batch_row_ind, const index_type* batch_col_ind,
																 const value_type* A, value_type beta);

void MatrixCSRSetValueBlockedBatchedGPU(value_type* matval, value_type alpha,
																				index_type csr_num_row, index_type csr_num_col,
																				const index_type* csr_row_ptr, const index_type* csr_col_ind,
																				index_type batch_size,
																				const index_type* batch_row_ind, const index_type* batch_col_ind,
																				index_type block_row, index_type block_col,
																				const value_type* A, value_type beta, int lda, int stride);

void SetBlockValueToSubmatGPU(value_type** matval, value_type alpha,
															index_type n_offset, const index_type* offset, index_type nshl,
															index_type batch_size, const index_type* batch_index_ptr, const index_type* ien,
															index_type num_row, index_type num_col,
															const index_type* row_ptr, const index_type* col_ind,
															const value_type* val, int lda, int stride, value_type beta, const index_type* mask);

__END_DECLS__


#endif /* __MATRIX_IMPL_H__ */
