#ifndef __MATRIX_IMPL_H__
#define __MATRIX_IMPL_H__

#include "matrix.h"

__BEGIN_DECLS__

void MatrixCSRGetDiagGPU(const value_type* val, const index_type* row_ptr, const index_type* col_ind, value_type* diag, index_type num_row); 


__END_DECLS__


#endif /* __MATRIX_IMPL_H__ */
