#ifndef __MATRIX_IMPL_H__
#define __MATRIX_IMPL_H__

#include "matrix.h"

__BEGIN_DECLS__

void MatrixCSRGetDiagGPU(const value_type* val, const index_type* row_ptr, const index_type* col_ind, value_type* diag, index_type num_row); 

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

__END_DECLS__


#endif /* __MATRIX_IMPL_H__ */
