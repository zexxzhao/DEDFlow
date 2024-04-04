#include "common.h"

__BEGIN_DECLS__

/*
 * y = x / diag(A)
 */

void PCJacobiDevice(u32 n, u32 nnz, f64* data, u32* row_ptr, u32* col_idx, f64* x, f64* y);
void PCJacobiInplaceDevice(u32 n, u32 nnz, f64* data, u32* row_ptr, u32* col_idx, f64* x);

__END_DECLS__
