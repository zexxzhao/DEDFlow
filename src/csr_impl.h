#ifndef __CSR_IMPL_H__
#define __CSR_IMPL_H__

#include "csr.h"

void ExpandCSRByBlockSize(const CSRAttr* attr, CSRAttr* new_attr, csr_index_type block_size[2]);

#endif /* __CSR_IMPL_H__ */
