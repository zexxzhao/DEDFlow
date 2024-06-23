#include "common.h"

__BEGIN_DECLS__

static __global__ void
GMRESUpdateResidualUpdateKernel(f64* beta, f64* gv) {
	f64 beta_new[2];
	beta_new[0] = beta[0];
	/* cos = gv[0], sin = gv[1] */
	beta_new[1] = -gv[1] * beta_new[0];
	beta_new[0] *= gv[0];
	beta[0] = beta_new[0];
	beta[1] = beta_new[1];
}


void GMRESResidualUpdatePrivate(f64* beta, f64* gv) {
	GMRESUpdateResidualUpdateKernel<<<1, 1>>>(beta, gv);
}

static __global__ void
MatrixGetDiagBlockKernel(const value_type* matval,
												 index_type block_size,
												 index_type num_row, index_type num_col,
												 const index_type* row_ptr, const index_type* col_idx,
												 value_type* diag_block, int lda, int stride) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num_row) return;

	index_type k, i, j;
	index_type row_start, row_end, len;

	row_start = row_ptr[idx];
	row_end = row_ptr[idx + 1];
	len = row_end - row_start;
	for(k = row_start; k < row_end; k++) {
		if(col_idx[k] == idx) {
			break;
		}
	}

	matval += row_start * block_size * block_size + (k - row_start) * block_size;
	diag_block += idx * stride;
	for(i = 0; i < block_size; i++) {
		for(j = 0; j < block_size; j++) {
			// if(stride <= i * lda + j) {
			// 	printf("idx = %d, i = %d, j = %d, lda = %d, stride = %d\n", idx, i, j, lda, stride);
			// }
			diag_block[i * lda + j] = matval[i * len * block_size + j];
			// diag_block[i * lda + j] = idx;
		}
	}
	// if(idx == 6 && diag_block[6] == 0 && diag_block[7] == 0 && diag_block[8] == 0) {
	// 	printf("idx = %d\n", idx);
	// 	for(i = 0; i < block_size; i++) {
	// 		for(j = 0; j < block_size; j++) {
	// 			printf("%f ", matval[i * len * block_size + j]);
	// 		}
	// 		printf("\n");
	// 	}
	// }
}

void MatrixGetDiagBlock(const value_type* matval,
												 index_type block_size,
												 index_type num_row, index_type num_col,
												 const index_type* row_ptr, const index_type* col_idx,
												 value_type* diag_block, int lda, int stride) {
	int num_thread = 256;
	int num_block = (num_row + num_thread - 1) / num_thread;
	MatrixGetDiagBlockKernel<<<num_block, num_thread>>>(matval, block_size, num_row, num_col, row_ptr, col_idx, diag_block, lda, stride);
}

__END_DECLS__
