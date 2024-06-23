#ifndef __PC_IMPL_H__
#define __PC_IMPL_H__

__BEGIN_DECLS__

void PCJacobiDevice(index_type n, index_type nnz, f64* data, index_type* row_ptr, index_type* col_idx, f64* x, f64* y);
void PCJacobiInplaceDevice(index_type n, index_type nnz, f64* data, index_type* row_ptr, index_type* col_idx, f64* x);



__END_DECLS__

#endif /* __PC_IMPL_H__ */
