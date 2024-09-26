#ifndef __CSR_IMPL_H__
#define __CSR_IMPL_H__

#include "csr.h"

#ifdef CDAM_USE_CUDA
__BEGIN_DECLS__
void ExpandCSRByBlockSizeDevice(const CSRAttr* attr, CSRAttr* new_attr, index_type block_size[2]);
void CSRAttrGetNZIndBatchedDevice(const CSRAttr* attr, index_type batch_size,
																	const index_type* row, const index_type* col,
																	index_type* ind);
__END_DECLS__
#endif

#endif /* __CSR_IMPL_H__ */
