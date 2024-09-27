#include "blas.h"
// #include "sequential_matrix.h"
#include "sequential_matrix_device.h"

#ifdef CDAM_USE_CUDA
#include <cub/cub.cuh>


__BEGIN_DECLS__
__global__ void SeqMatZeroRowDenseRowMajorKernel(value_type* data, index_type nrow, index_type ncol,
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
__global__ void SeqMatZeroRowDenseColMajorKernel(value_type* data, index_type nrow, index_type ncol,
																								 index_type nr, index_type* irow,
																								 index_type rshift, index_type cshift,
																								 value_type diag) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > ncol * nr) return;
	int ir = irow[i % nr] + rshift;
	int ic = i / nr + cshift;
	if (ir < 0 || ir >= nrow || ic < 0 || ic >= ncol) return;

	data[ir + ic * nrow] = (ir == ic) * diag;
}
void SeqMatZeroRowDenseGPU(value_type* data, index_type nrow, index_type ncol,
													 index_type nr, index_type* row,
													 index_type rshift, index_type cshift,
													 value_type diag, cudaStream_t stream) {
	int num_threads = 256;
	int num_blocks = CEIL_DIV(nr * ncol, num_threads);
	SeqMatZeroRowDenseRowMajorKernel<<<num_blocks, num_threads, 0, stream>>>(data, nrow, ncol, nr, row, rshift, cshift, diag);
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

void SeqMatGetSubmatDenseGPU(value_type* src, index_type nrow, index_type ncol,
														 index_type nr, index_type nc, index_type *row, index_type* col,
														 index_type rshift, index_type cshift, value_type* dst, cudaStream_t stream) {
	int num_threads = 256;
	int num_blocks = CEIL_DIV(nr * nc, num_threads);
	SeqMatGetSubmatDenseRowMajorKernel<<<num_blocks, num_threads, 0, stream>>>(src, nrow, ncol, dst, nr, nc, row, col);
}

__global__ void SeqMatGetDiagDenseKernel(value_type* data, index_type n, value_type* diag, index_type bs) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n * bs) return;

	int ir = i / bs;
	int ic = i % bs;

	diag[i] = data[ir * n + (ir / bs) * bs + ic]; 
}

void SeqMatGetDiagDenseGPU(value_type* data, index_type n,
													 value_type* diag, index_type bs, cudaStream_t stream) {
	if(bs == 1) {
		dcopy(n, data, n + 1, diag, 1);
	}
	else if(bs > 1) {
		int num_threads = 256;
		int num_blocks = CEIL_DIV(n * bs, num_threads);
		SeqMatGetDiagDenseKernel<<<num_blocks, num_threads, 0, stream>>>(data, n, diag, bs);
	}
}

__global__ void SeqMatAddElemValueBatchedDenseKernel(value_type* data, index_type nrow, index_type ncol,
																										 CdamLayout* cmap, CdamLayout* rmap,
																										 byte* row_mask, byte* col_mask,
																										 index_type nelem, index_type nshl, index_type* ien,
																										 value_type* value) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	index_type i, r, c, iel;
	index_type batch, ishl, jshl, ishg, jshg;
	index_type dst_row, dst_col;
	index_type block_row = CdamLayoutComponentOffset(rmap)[CdamLayoutNumComponent(rmap)];
	index_type block_col = CdamLayoutComponentOffset(cmap)[CdamLayoutNumComponent(cmap)];

	if (idx >= nelem * nshl * nshl) return;

	iel = idx / (nshl * nshl);
	ishl = (idx % (nshl * nshl)) / nshl;
	jshl = idx % nshl;

	if(row_mask[iel] & (1 << ishl) == 0) return;
	if(col_mask[iel] & (1 << jshl) == 0) return;

	ishg = ien[iel * nshl + ishl];
	jshg = ien[iel * nshl + jshl];
	

	value += idx * nshl * nshl * block_row * block_col;
	for(r = 0; r < block_row; r++) {
		dst_row = CdamLayoutDOFMapLocal(rmap, ishg, r);
		for(c = 0; c < block_col; ++c) {
			dst_col = CdamLayoutDOFMapLocal(cmap, jshg, c);
			data[dst_row * ncol + dst_col] += value[r * block_col + c];
		}
	}
}
__global__ void GenerateBatchedIndexMapKernel(index_type batch_size, index_type* batch_index_ptr,
																							index_type* ien, index_type nshl,
																							index_type nnode, index_type* comp_offset,
																							index_type block_displ, index_type block_count,
																							index_type* index_map) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > batch_size * nshl) return;
	int iel = batch_index_ptr[i / nshl];
	int ishl = i % nshl;
	int node = ien[iel * nshl + ishl];

	index_map += i * block_count;
	int j, k = 0;
	index_type begin, end;
	for(j = 0; j < block_count; ++j) {
		while(j + block_displ < comp_offset[k] && comp_offset[k + 1] >= comp_offset[k]) {
			k++;
		}
		begin = comp_offset[k];
		end = comp_offset[k + 1];
		index_map[j] = begin * nnode + node * (end - begin) + j + block_displ - begin;
	}
}
__global__ void SeqMatAddValueBatchedDenseKernel(value_type* data, index_type nrow, index_type ncol,
																								 index_type batch_size, index_type nshl,
																								 index_type* row_index_map, index_type* col_index_map,
																								 index_type block_row_count, index_type block_col_count,
																								 value_type* value, index_type ldv, index_type stride) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > batch_size * nshl * nshl) return; 

	index_type ir, ic, iel, ishl, jshl, ishg, jshg;
	index_type dst_row, dst_col;
	value_type* val = value + i * stride;

	iel = i / (nshl * nshl);
	ishl = (i % (nshl * nshl)) / nshl;
	jshl = i % nshl;

	row_index_map += (iel * nshl + ishl) * block_row_count;
	col_index_map += (iel * nshl + jshl) * block_col_count;

	for(ir = 0; ir < block_row_count; ++ir) {
		dst_row = row_index_map[ir];
		for(ic = 0; ic < block_col_count; ++ic) {
			dst_col = col_index_map[ic];
			data[dst_row * ncol + dst_col] += val[ir * ldv + ic];
		}
	}
}



// void SeqMatAddElemValueBatchedDenseGPU(value_type* data, index_type nrow, index_type ncol,
// 																			 CdamLayout* cmap, CdamLayout* rmap,
// 																			 index_type batch_size, index_type* batch_index_ptr,
// 																			 index_type* ien, index_type nshl,
// 																			 index_type block_row_displ, index_type block_row_count,
// 																			 index_type block_col_displ, index_type block_col_count,
// 																			 value_type* value, index_type ldv, index_type stride,
// 																			 cudaStream_t stream, Arena scratch) {
// 	int num_threads = 256;
// 	int num_blocks = CEIL_DIV(batch_size * nshl * nshl, num_threads);
// 
// 
// 	/* Generate the row index map of the batch */
// 	index_type rnode = CdamLayoutNumOwned(rmap) + CdamLayoutNumGhosted(rmap);
// 	index_type rcomp_offset = CdamLayoutComponentOffset(rmap);
// 	index_type* row_index_map = (index_type*)ArenaPush(sizeof(index_type), batch_size * nshl * 6, &scratch, ARENA_FLAG_NONZERO);
// 	GenerateBatchedIndexMapKernel<<<CEIL_DIV(batch_size * nshl, num_threads), num_threads, 0, stream>>>(batch_size, batch_index_ptr, ien, nshl, rnode, rcomp_offset, block_row_displ, block_row_count, row_index_map);
// 	/* Generate the column index map of the batch */
// 	index_type cnode = CdamLayoutNumOwned(cmap) + CdamLayoutNumGhosted(cmap);
// 	index_type ccomp_offset = CdamLayoutComponentOffset(cmap);
// 	index_type* col_index_map = (index_type*)ArenaPush(sizeof(index_type), batch_size * nshl * 6, &scratch, ARENA_FLAG_NONZERO);
// 	GenerateBatchedIndexMapKernel<<<CEIL_DIV(batch_size * nshl, num_threads), num_threads, 0, stream>>>(batch_size, batch_index_ptr, ien, nshl, cnode, ccomp_offset, block_col_displ, block_col_count, col_index_map);
// 	/* Add the value to the data */
// 	SeqMatAddValueBatchedDenseKernel<<<num_blocks, num_threads, 0, stream>>>(data, nrow, ncol, batch_size, nshl, row_index_map, col_index_map, block_row_count, block_col_count, value, ldv, stride);
// }

void SeqMatAddElemValueBatchedDenseGPU(value_type* data, index_type nrow, index_type ncol, 
																			 CdamLayout* rmap, CdamLayout* cmap,
																			 byte* row_mask, byte* col_mask,
																			 index_type nelem, index_type nshl, index_type* ien,
																			 value_type* value, Arena scratch, cudaStream_t stream) {
	int num_threads = 256;
	int num_blocks = CEIL_DIV(nelem * nshl * nshl, num_threads);

	SeqMatAddElemValueBatchedDenseKernel<<<num_blocks, num_threads, 0, stream>>>(data, nrow, ncol, cmap, rmap, row_mask, col_mask, nelem, nshl, ien, value);
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
	index_type nrow = CSRAttrNumRow(spy);
	index_type ncol = CSRAttrNumCol(spy);
	int num_threads = 256;
	int num_blocks = CEIL_DIV(nrow, num_threads);
	SeqMatGetDiagBlockedCSRKernel<<<num_blocks, num_threads, 0, stream>>>(src, nrow, ncol, CSRAttrRowPtr(spy), CSRAttrColInd(spy), diag, ld, stride, bs);
}


__global__ void SeqMatAddElemValueCSRKernel(value_type* matval, index_type nrow, index_type ncol,
																						index_type* row_ptr, index_type* col_ind,
																						CdamLayout* rmap, CdamLayout* cmap,
																						byte* row_mask, byte* col_mask,
																						index_type nelem, index_type nshl, index_type* ien,
																						value_type* value) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= nelem * nshl * nshl) return;
	index_type iel = i / (nshl * nshl);
	index_type ishl = (i % (nshl * nshl)) / nshl;
	index_type jshl = i % nshl;
	index_type ishg, jshg;

	index_type block_row = CdamLayoutComponentOffset(rmap)[CdamLayoutNumComponent(rmap)];
	index_type block_col = CdamLayoutComponentOffset(cmap)[CdamLayoutNumComponent(cmap)];

	index_type start, end, len;

	index_type dst_row, dst_col;
	index_type r, c, k;

	if(row_mask[iel] & (1 << ishl) == 0) return;
	if(col_mask[iel] & (1 << jshl) == 0) return;

	ishg = ien[iel * nshl + ishl];
	jshg = ien[iel * nshl + jshl];


	value += i * nshl * nshl * block_row * block_col;

	for(r = 0; r < block_row; ++r) {
		dst_row = CdamLayoutDOFMapLocal(rmap, ishg, r);
		start = row_ptr[dst_row];
		end = row_ptr[dst_row + 1];
		len = end - start;

		k = start;
		for(c = 0; c < block_col; ++c) {
			dst_col = CdamLayoutDOFMapLocal(cmap, jshg, c);

			for(; k < end; ++k) {
				if(col_ind[k] == dst_col) {
					matval[k] += value[r * block_col + c];
					break;
				}
			}
		}
	}

}

void SeqMatAddElemValueCSRGPU(value_type* matval,
															CSRAttr* spy, CdamLayout* rmap, CdamLayout* cmap,
															byte* row_mask, byte* col_mask,
															index_type nelem, index_type nshl, index_type* ien,
															value_type* val, Arena scratch, cudaStream_t stream) {
	index_type nrow = CSRAttrNumRow(spy);
	index_type ncol = CSRAttrNumCol(spy);
	int num_threads = 256;
	int num_blocks = CEIL_DIV(nelem * nshl * nshl, num_threads);

	SeqMatAddElemValueCSRKernel<<<num_blocks, num_threads, 0, stream>>>(
		matval, nrow, ncol, CSRAttrRowPtr(spy), CSRAttrColInd(spy), rmap, cmap, row_mask, col_mask, nelem, nshl, ien, val);
}

__END_DECLS__
#endif /* CDAM_USE_CUDA */
