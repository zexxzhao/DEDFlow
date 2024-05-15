#ifndef __MATRIX_IMPL_H__
#define __MATRIX_IMPL_H__

#include "matrix.h"

__BEGIN_DECLS__

void MatrixCSRGetDiagGPU(const value_type* val, const index_type* row_ptr, const index_type* col_ind, value_type* diag, index_type num_row); 

void MatrixCSRAddElementLHSGPU(value_type* matval,
															const index_type* row_ptr, const index_type* col_ind, index_type num_row, index_type num_col,
															index_type nshl, index_type bs, index_type num_batch, const index_type* ien, const index_type* batch_ptr,
															const value_type* val);


__END_DECLS__


#endif /* __MATRIX_IMPL_H__ */
