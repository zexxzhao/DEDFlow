#include "sequential_matrix.h"

#ifdef CDAM_USE_CUDA


__BEGIN_DECLS__
__global__ void SeqMatZeroRowSparseKernel(value_type* data, index_type nrow, index_type ncol,
																					index_type nr, index_type* irow,
																					index_type rshift, index_type cshift,
																					value_type diag) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > ncol * nr) return;
	int ir = irow[i / ncol] + rshift;
	int ic = i % ncol + cshift;
	if (ir < 0 || ir >= nrow || ic < 0 || ic >= ncol) return;

	data[ir * ncol + ic] = (ir == ic) * diag; 
}
void SeqMatZeroRowDenseGPU(value_type* data, index_type nrow, index_type ncol,
													 index_type nr, index_type* row,
													 index_type rshift, index_type cshift,
													 value_type diag, cudaStream_t stream) {
	int num_threads = 256;
	int num_blocks = CEIL_DIV(n * ncol, num_threads);
	if(nrow != ncol)
		diag = 0.0;
	SeqMatZeroRowDenseKernel<<<num_blocks, num_threads, 0, stream>>>(data, nrow, ncol, n, row, rshift, cshift, diag);
}
__END_DECLS__
#endif /* CDAM_USE_CUDA */
