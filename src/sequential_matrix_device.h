#ifndef __SEQUENTIAL_MATRIX_DEVICE_H__
#define __SEQUENTIAL_MATRIX_DEVICE_H__

#include "common.h"

#ifdef CDAM_USE_CUDA
#include <cuda_runtime.h>

__BEGIN_DECLS__
void SeqMatZeroRowDenseGPU(value_type* data, index_type nrow, index_type ncol,
													 index_type nr, index_type* row,
													 index_type rshift, index_type cshift,
													 value_type diag, cudaStream_t stream);
__END_DECLS__

#endif /* CDAM_USE_CUDA */
#endif /* __SEQUENTIAL_MATRIX_DEVICE_H__ */
