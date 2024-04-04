#include "common.h"

__BEGIN_DECLS__

__global__ __device__ void
PCJacobiApplyKernel(u32 n, u32 nnz, f64* data, u32* row_ptr, u32* col_idx, f64* x, f64* y);
__global__ __device__ void
PCJacobApplyInPlaceKernel(u32 n, u32 nnz, f64* data, u32* row_ptr, u32* col_idx, f64* x);

__END_DECLS__
