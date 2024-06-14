#ifndef __CSR_IMPL_H__
#define __CSR_IMPL_H__

#include "csr.h"

void ExpandCSRByBlockSize(const CSRAttr* attr, CSRAttr* new_attr, csr_index_type block_size[2]);
void CSRAttrGetNZIndBatchedGPU(const CSRAttr* attr, csr_index_type batch_size,
															 const index_type* row, const index_type* col,
															 csr_index_type* ind);

#endif /* __CSR_IMPL_H__ */
