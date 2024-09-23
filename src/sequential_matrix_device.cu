#include "blas.h"
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
__global__ void SeqMatGetSubmatDenseColMajorKernel(value_type* src, index_type nrow, index_type ncol,
																									 value_type* dst, index_type nr, index_type nc,
																									 index_type* row, index_type* col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nr * nc) return;
	int ir = row[i % nr];
	int ic = col[i / nr];
	if (ir < 0 || ir >= nrow || ic < 0 || ic >= ncol) return;

	dst[i] = src[ir + ic * nrow];
}
__global__ void SeqMatGetSubmatDenseRowMajorKernel(value_type* src, index_type nrow, index_type ncol,
																									 value_type* dst, index_type nr, index_type nc,
																									 index_type* row, index_type* col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nr * nc) return;
	int ir = row[i / nc];
	int ic = col[i % nc];
	if (ir < 0 || ir >= nrow || ic < 0 || ic >= ncol) return;

	dst[i] = src[ir * ncol + ic];
}

void SeqMatGetSubmatDenseGPU(MatOrder order, value_type* src, index_type nrow, index_type ncol,
														 value_type* dst, index_type nr, index_type nc,
														 index_type* row, index_type* col, cudaStream_t stream) {
	int num_threads = 256;
	int num_blocks = CEIL_DIV(nr * nc, num_threads);
	if (order == ROW_MAJOR) {
		SeqMatGetSubmatDenseRowMajorKernel<<<num_blocks, num_threads, 0, stream>>>(src, nrow, ncol, dst, nr, nc, row, col);
	} else {
		SeqMatGetSubmatDenseColMajorKernel<<<num_blocks, num_threads, 0, stream>>>(src, nrow, ncol, dst, nr, nc, row, col);
	}
}

__global__ void SeqMatGetDiagDenseKernel(value_type* data, index_type n, value_type* diag, index_type bs) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= k * bs * bs) return;

	int ir = i / bs;
	int ic = i % bs;

	diag[i] = data[ir * n + (ir / bs) * bs + ic]; 
}

void SeqMatGetDiagDenseGPU(value_type* data, index_type n,
													 value_type* diag, index_type bs, cudaStream_t stream) {
	if(bs == 1) {
		dcopy(n, data, n + 1, diag, 1);
	}
	else (bs > 1) {
		int num_threads = 256;
		int num_blocks = CEIL_DIV(n * bs, num_threads);
		SeqMatGetDiagDenseKernel<<<num_blocks, num_threads, 0, stream>>>(data, n, diag, bs);
	}
}

__global__ void SeqMatAddElemValueBatchedDenseKernel(value_type* data, index_type nrow, index_type ncol,
																										 MatOrder order,
																										 MatStorageMethod rmap_storage, MatStorageMethod cmap_storage,
																										 index_type batch_size, index_type* batch_index_ptr,
																										 index_type* ien, index_type nshl,
																										 index_type block_row, index_type block_col,
																										 value_type* value, index_type ldv, index_type stride) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	index_type i, ir, ic, iel;
	index_type batch, ishl, jshl; ishg, jshg;
	index_type dst_row, dst_col;

	if (idx >= batch_size * nshl * nshl) return;

	iel = idx / (nshl * nshl);
	iel = batch_index_ptr[iel];
	ishl = (idx % (nshl * nshl)) / nshl;
	jshl = idx % nshl;

	ishg = ien[iel * nshl + ishl];
	jshg = ien[iel * nshl + jshl];

	value += idx * stride;
	for(ir = 0; i < block_row; i++) {
		dst_row = DOFMap(rmap_storage, nrow / block_row, block_row, ishg, ir);
		for(ic = 0; ic < block_col; ++ic) {
			dst_col = DOFMap(cmap_storage, ncol / block_col, block_col, jshg, ic);
			data[DOFMap(order, nrow, ncol, dst_row, dst_col)] += value[ir * ldv + ic];
		}
	}
}

void SeqMatAddElemValueBatchedDenseGPU(value_type* data, index_type nrow, index_type ncol,
																			 MatOrder order,
																			 MatStorageMethod rmap_storage, MatStorageMethod cmap_storage,
																			 index_type batch_size, index_type* batch_index_ptr,
																			 index_type* ien, index_type nshl,
																			 index_type block_row, index_type block_col,
																			 value_type* value, index_type ldv, index_type stride,
																			 cudaStream_t stream) {
	int num_threads = 256;
	int num_blocks = CEIL_DIV(batch_size * nshl * nshl, num_threads);
	SeqMatAddElemValueBatchedDenseKernel<<<num_blocks, num_threads, 0, stream>>>(data, nrow, ncol, order, rmap_storage, cmap_storage, batch_size, batch_index_ptr, ien, nshl, block_row, block_col, value, ldv, stride);
}

__global__ void SeqMatCopySubmatValueCSRKernel(value_type* src, index_type* row_ptr, index_type* col_ind,
																							 index_type nr, index_type* row, index_type nc, index_type* col,
																							 value_type* dst, index_type* dst_row_ptr) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= nr) return;
	int j, k, ir, ic;

	index_type start = row_ptr[row[i]];
	index_type end = row_ptr[row[i] + 1];

	dst += dst_row_ptr[i];

	for(j = start; j < end; ++j) {
		for(k = 0; k < nc; ++k) {
			if(col_ind[j] == col[k]) {
				dst[k] = src[j];
			}
		}
	}
}

void SeqMatCopySubmatValueCSRGPU(value_type* src, CSRAttr* src_spy,
																 index_type nr, index_type* row, index_type nc, index_type* col,
																 value_type* dst, CSRAttr* dst_spy, cudaStream_t stream) {
	int num_threads = 256;
	int num_blocks = CEIL_DIV(nr, num_threads);
	index_type* row_ptr = CSRAttrRowPtr(src_spy);
	index_type* col_ind = CSRAttrColInd(src_spy);
	index_type* dst_row_ptr = CSRAttrRowPtr(dst_spy);
	SeqMatCopySubmatValueCSRKernel<<<num_blocks, num_threads, 0, stream>>>(src, row_ptr, col_ind, nr, row, nc, col, dst, dst_row_ptr);
}

__global__ void SeqMatGetDiagBlockedCSRKernel(value_type* src, index_type nrow, index_type ncol,
																							index_type* row_ptr, index_type* col_ind,
																							index_type* diag, index_type ld, index_type stride, index_type bs) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= nrow) return;
	int j, k, start, end, ir, ic;
	int ib, lane, range[2];
	start = row_ptr[i];
	end = row_ptr[i + 1];
	ib = i / bs;
	lane = i % bs;
	range[0] = ib * bs;
	range[1] = MIN(range[0] + bs, ncol);

	src += start;
	diag += ib * stride + lane * ld; 

	for(j = start; j < end; ++j) {
		ic = col_ind[j];
		if(ic >= range[0] && ic < range[1]) {
			diag[ic - range[0]] = src[j];
		}
	}
}

void SeqMatGetDiagBlockedCSRGPU(value_type* src, CSRAttr* spy, 
																index_type* diag, index_type ld, index_type stride, index_type bs,
																cudaStream_t stream) {
	int num_threads = 256;
	int num_blocks = CEIL_DIV(nrow, num_threads);
	SeqMatGetDiagBlockedCSRKernel<<<num_blocks, num_threads, 0, stream>>>(src, nrow, ncol, CSRAttrRowPtr(spy), CSRAttrColInd(spy), diag, ld, stride, bs);
}


__global__ void SeqMatAddElemValueCSRKernel(value_type alpha, value_type* matval, 
																						index_type batch_size, index_type* batch_index_ptr,
																						index_type* ien, index_type nshl,
																						index_type nrow, index_type ncol,
																						index_type* row_ptr, index_type* col_ind,
																						MatStorageMethod rmap_storage, MatStorageMethod cmap_storage,
																						index_type block_row, index_type block_col,
																						value_type beta, value_type* val, int lda, int stride) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= batch_size * nshl * nshl) return;
	index_type iel = batch_index_ptr[i / (nshl * nshl)];
	index_type ishl = (i % (nshl * nshl)) / nshl;
	index_type jshl = i % nshl;

	index_type ishg = ien[iel * nshl + ishl];
	index_type jshg = ien[iel * nshl + jshl];

	index_type start, end, len;

	index_type dst_row, dst_col;
	index_type ii, jj, k;

	val += i * stride;

	for(ii = 0; ii < block_row; ++ii) {
		dst_row = DOFMap(rmap_storage, nrow / block_row, block_row, ishg, ii);

		start = row_ptr[dst_row];
		end = row_ptr[dst_row + 1];
		len = end - start;

		for(jj = 0; jj < block_col; ++jj) {
			dst_col = DOFMap(cmap_storage, ncol / block_col, block_col, jshg, jj);

			for(k = start; k < end; ++k) {
				if(col_ind[k] == dst_col) {
					matval[k] = alpha * matval[k] + beta * val[ii * lda + jj];
				}
			}
		}
	}

}

void SeqMatAddElemValueCSRGPU(value_type alpha, value_type* matval,
															CSRAttr* spy, MatStorageMethod rmap_storage, MatStorageMethod cmap_storage,
															index_type batch_size, index_type* batch_index_ptr,
															index_type* ien, index_type nshl,
															index_type block_row, index_type block_col,
															value_type beta, value_type* val, int lda, int stride, cudaStream_t stream) {
	int num_threads = 256;
	int num_blocks = CEIL_DIV(batch_size * nshl * nshl, num_threads);

	SeqMatAddElemValueCSRKernel<<<num_blocks, num_threads, 0, stream>>>(
			alpha, matval, batch_size, batch_index_ptr, ien, nshl,
			CSRAttrNumRow(spy), CSRAttrNumCol(spy), CSRAttrRowPtr(spy), CSRAttrColInd(spy),
			rmap_storage, cmap_storage, block_row, block_col, beta, val, lda, stride);
}
__END_DECLS__
#endif /* CDAM_USE_CUDA */
