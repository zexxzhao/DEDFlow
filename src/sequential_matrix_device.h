#ifndef __SEQUENTIAL_MATRIX_DEVICE_H__
#define __SEQUENTIAL_MATRIX_DEVICE_H__

#include "common.h"
#include "alloc.h"
#include "layout.h"
#include "csr.h"
#include "matrix_util.h"

#ifdef CDAM_USE_CUDA
#include <cuda_runtime.h>

__BEGIN_DECLS__
void SeqMatZeroRowDenseGPU(value_type* data, index_type nrow, index_type ncol,
													 index_type nr, index_type* row,
													 index_type rshift, index_type cshift,
													 value_type diag, cudaStream_t stream);

void SeqMatGetSubmatDenseGPU(value_type* data, index_type nrow, index_type ncol,
														 index_type nr, index_type nc, index_type* row, index_type* col,
														 index_type rshift, index_type cshift, value_type* submat, cudaStream_t stream);

void SeqMatGetDiagDenseGPU(value_type* data, index_type n,
													 value_type* diag, index_type bs, cudaStream_t stream);

void SeqMatAddElemValueBatchedDenseGPU(value_type* data, index_type nrow, index_type ncol,
																			 CdamLayout* rmap, CdamLayout* cmap,
																			 byte* row_mask, byte* col_mask,
																			 index_type nelem, index_type nshl, index_type* ien,
																			 value_type* value, Arena scratch, cudaStream_t stream);

void SeqMatZeroRowCSRGPU(value_type* matval, CSRAttr* spy,
												 index_type nr, index_type* row,
												 index_type shift, value_type diag, cudaStream_t stream);
void SeqMatCopySubmatValueCSRGPU(value_type* src, CSRAttr* src_spy,
																 index_type nr, index_type* row, index_type nc, index_type* col,
																 value_type* dst, CSRAttr* dst_spy, cudaStream_t stream);


void SeqMatGetDiagBlockedCSRGPU(value_type* src, CSRAttr* spy, 
																value_type* diag, index_type ld, index_type stride,
																index_type bs, cudaStream_t stream);
void SeqMatAddElemValueBatchedCSRGPU(value_type* matval, CSRAttr* spy,
																		 CdamLayout* rmap, CdamLayout* cmap,
																		 byte* row_mask, byte* col_mask,
																		 index_type elem, index_type nshl, index_type* ien,
																		 value_type* val, Arena scratch, cudaStream_t stream);


__END_DECLS__

#endif /* CDAM_USE_CUDA */
#endif /* __SEQUENTIAL_MATRIX_DEVICE_H__ */
