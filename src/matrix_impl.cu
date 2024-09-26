#include <assert.h>
#include "alloc.h"

#ifdef CDAM_USE_CUDA 


template <typename I, typename T>
__global__ void
MatrixCSRZeroRowKernel(T* matval,
											 I num_row, I num_col, const I* row_ptr, const I* col_ind,
											 I n, const I* row, I shift, T diag) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= n) {
		return;
	}
	I ir = shift + row[i];
	if(ir < 0 || ir >= num_row) {
		return;
	}
	I start = row_ptr[ir], end = row_ptr[ir + 1];
	for(I j = start; j < end; j++) {
		matval[j] = diag * static_cast<T>(col_ind[j] == ir);
	}
}

template <typename ValueType, typename IndexType>
__global__ void
MatrixCSRGetDiagKernel(const ValueType* val, const IndexType* row_ptr, const IndexType* col_idx, ValueType* diag, const IndexType num_row) {
	const IndexType i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= num_row) {
		return;
	}
	b32 found = FALSE;
	IndexType start = row_ptr[i], end = row_ptr[i + 1];
	for(IndexType j = start; j < end; j++) {
		if(col_idx[j] == i) {
			diag[i] = val[j];
			// if(diag[i] == 0.0) {
			// 	printf("i=%d, col_idx[%d]=%d, val[%d]=%g\n", i, j, col_idx[j], j, val[j]);
			// }
			found = TRUE;
			break;
		}
	}
}

template <typename I>
__device__ I CSRFindIndex(I begin, I end, const I* col_ind, I col) {
	for(; begin < end; begin++) {
		if(col_ind[begin] == col) {
			return begin;
		}
	}
	return (I)(-1);
}

template <typename I, typename T>
__global__ void
MatrixCSRSetValuesCOOKernel(T* matval, T alpha,
													  I num_row, I num_col,
													  const I* row_ptr, const I* col_ind,
													  I n, const I* row, const I* col,
													  const T* val, T beta) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= n) return;
	I ir = row[i];
	I k = CSRFindIndex(row_ptr[ir], row_ptr[ir + 1], col_ind, col[i]);
	if(k != (I)(-1)) {
		matval[k] = alpha * matval[k] + beta * val[i];
	}
	// for(I k = start; k < end; ++k) {
	// 	if(col_ind[k] == ic) {
	// 		matval[k] = alpha * matval[i] + beta * val[i];
	// 	}
	// }
}

template <typename I, typename T>
__global__ void
MatrixCSRSetValuesIndKernel(T* matval, T alpha,
														I n, const I* ind,
														const T* val, T beta) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= n) return;
	matval[i] = alpha * matval[i] + beta * val[i];
}


template <typename I, typename T>
__global__ void
MatrixCSRAddElemValueBatchedKernel(T* matval, T alpha,
																	 I batch_size, const I* batch_index_ptr,
																	 const I* ien, I nshl,
																	 I num_row, I num_col, const I* row_ptr, const I* col_ind,
																	 const value_type* val, T beta) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= batch_size * nshl * nshl) return;
	I iel = batch_index_ptr[i / (nshl * nshl)];
	I row = iel % (nshl * nshl) / nshl;
	I col = iel % nshl;
	row = ien[iel * nshl + row];
	col = ien[iel * nshl + col];
	I k;
	for(k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
		if(col_ind[k] == col) {
			break;
		}
	}
	matval[k] = alpha * matval[k] + beta * val[i];
}
template <typename I, typename T>
__global__ void
MatrixCSRAddElemValueBatchedMaskedKernel(T* matval, T alpha,
																	 I batch_size, const I* batch_index_ptr,
																	 const I* ien, I nshl,
																	 I num_row, I num_col, const I* row_ptr, const I* col_ind,
																	 const value_type* val, T beta, const I* mask) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= batch_size * nshl * nshl) return;
	if(mask[i] == 0) return;
	I iel = batch_index_ptr[i / (nshl * nshl)];
	I row = iel % (nshl * nshl) / nshl;
	I col = iel % nshl;
	row = ien[iel * nshl + row];
	col = ien[iel * nshl + col];
	I k;
	for(k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
		if(col_ind[k] == col) {
			break;
		}
	}
	matval[k] = alpha * matval[k] + beta * val[i];
}

template <typename I, typename T>
__global__ void
MatrixCSRAddElemValueBlockedBatchedKernel(T* matval, T alpha,
																					I batch_size, const I* batch_index_ptr, const I* ien, I nshl,
																					I num_row, I num_col, const I* row_ptr, const I* col_ind,
																					I block_row, I block_col,
																					const T* val, int lda, int stride, T beta) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= batch_size * nshl * nshl) return;
	I iel = batch_index_ptr[i / (nshl * nshl)];
	I row = iel % (nshl * nshl) / nshl;
	I col = iel % nshl;
	row = ien[iel * nshl + row];
	col = ien[iel * nshl + col];
	I start = row_ptr[row], end = row_ptr[row + 1];
	I len = end - start;
	I k;
	for(k = start; k < end; ++k) {
		if(col_ind[k] == col) {
			break;
		}
	}

	//matval += k * block_row * block_col;
	matval += k;
	val += i * stride;

	for(I ii = 0; ii < block_row; ++ii) {
		for(I jj = 0; jj < block_col; ++jj) {
			matval[len * block_col * ii + jj] = alpha * matval[len * block_col * ii + jj] 
																				+ beta * val[ii * lda + jj];
		}
	}
}

template <typename I, typename T>
__global__ void
MatrixCSRAddElemValueBlockedBatchedMaskedKernel(T* matval, T alpha,
																								I batch_size, const I* batch_index_ptr, const I* ien, I nshl,
																								I num_row, I num_col, const I* row_ptr, const I* col_ind,
																								I block_row, I block_col,
																								const T* val, int lda, int stride, T beta, const I* mask) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= batch_size * nshl * nshl) return;
	if(mask[i] == 0) return;
	I iel = batch_index_ptr[i / (nshl * nshl)];
	I row = iel % (nshl * nshl) / nshl;
	I col = iel % nshl;
	row = ien[iel * nshl + row];
	col = ien[iel * nshl + col];
	I start = row_ptr[row], end = row_ptr[row + 1];
	I len = end - start;
	I k;
	for(k = start; k < end; ++k) {
		if(col_ind[k] == col) {
			break;
		}
	}

	matval += k * block_row * block_col;
	val += i * stride;

	for(I ii = 0; ii < block_row; ++ii) {
		for(I jj = 0; jj < block_col; ++jj) {
			matval[len * block_col * ii + jj] = alpha * matval[len * block_col * ii + jj] 
																				+ beta * val[ii * lda + jj];
		}
	}
}

template <typename IndexType, typename ValueType>
__global__ void
MatrixCSRAddElementLHSKernel(ValueType* matval, IndexType NSHL, IndexType BS,
														 IndexType num_row, const IndexType* row_ptr,
														 IndexType num_col, const IndexType* col_ind,
														 IndexType batch_size, const IndexType* batch_ptr, const IndexType* ien,
														 const ValueType* val, int lda) {
	const IndexType idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= batch_size) return;
	IndexType iel = batch_ptr[idx];
	val += NSHL * NSHL * BS * BS * idx;

	IndexType vert[8];
	for(IndexType aa = 0; aa < NSHL; ++aa) {
		vert[aa] = ien[iel * NSHL + aa];
	}
	
	for(IndexType aa = 0; aa < NSHL; ++aa) {
		for(IndexType ii = 0; ii < BS; ++ii) {
			IndexType ir = vert[aa] * BS + ii;
			IndexType row_start = row_ptr[ir];
			IndexType row_end = row_ptr[ir + 1];
			for(IndexType bb = 0; bb < NSHL; ++bb) {
				for(IndexType jj = 0; jj < BS; ++jj) {
					IndexType ic = vert[bb] * BS + jj;
					b32 found = FALSE;
					for(IndexType j = row_start; j < row_end; j++) {
						if(col_ind[j] == ic) {
							matval[j] += val[(aa * BS + ii) * BS * NSHL + bb * BS + jj];
							found = TRUE;
							break;
						}
					}
				}
			}
		}
	}
	
}


/** 
 * @brief Add elementwise jacobian matrices into the global matrix
 * 
 * @tparam ValueType The matrix value type
 * @tparam IndexType The matrix index type
 * @param matval 
 * @param n_offset 
 * @param offset 
 * @param NSHL 
 * @param BS
 * @param num_row 
 * @param row_ptr 
 * @param num_col 
 * @param col_ind 
 * @param batch_size 
 * @param batch_ptr 
 * @param ien 
 * @param val The shape is (BS, BS, *). The last dimension is the batch_size * NSHL * NSHL
 * @param stride The stride of the last dimension of val
 */

template <typename MatrixType, typename ValueType, typename IndexType>
__global__ void
MatrixFSAddElementLHSKernel(MatrixType* mat[], IndexType n_offset, const IndexType* offset,
																IndexType NSHL,
																IndexType num_row, const IndexType* row_ptr,
																IndexType num_col, const IndexType* col_ind,
																IndexType batch_size, const IndexType* batch_ptr, const IndexType* ien,
																const ValueType* val, int lda) {
	const IndexType idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= NSHL * NSHL * batch_size) return;	
	IndexType iel = batch_ptr[idx / (NSHL * NSHL)];
	IndexType aa = idx % (NSHL * NSHL) / NSHL;
	IndexType bb = idx % NSHL;
	val += lda * lda * idx;
	IndexType row = ien[iel * NSHL + aa];
	IndexType col = ien[iel * NSHL + bb];

	IndexType row_start = row_ptr[row];
	IndexType row_end = row_ptr[row + 1];
	IndexType c;
	for(c = row_start; c < row_end; c++) {
		if(col_ind[c] == col) {
			break;
		}
	}
	for(IndexType i = 0; i < n_offset; i++) {
		for(IndexType j = 0; j < n_offset; j++) {
			if(mat[i * n_offset + j]->type != MAT_TYPE_CSR) {
				continue;
			}
			MatrixCSR* m = (MatrixCSR*)mat[i * n_offset + j]->data;
			IndexType kn = offset[i + 1] - offset[i], ln = offset[j + 1] - offset[j];
			for(IndexType k = 0; k < kn; ++k) {
				for(IndexType l = 0; l < ln; ++l) {
					m->val[row_start * kn * ln + k * (row_end - row_start) * ln + c * ln + l] += 
						val[(k + offset[i]) * lda + offset[j] + l];
				}
			}
		}
	}
	
}


template <typename I, typename T>
__global__ void
MatrixCSRSetValueBatchedKernel(T* matval, T alpha,
															 I csr_num_row, I csr_num_col,
															 const I* csr_row_ptr, const I* csr_col_ind,
															 I batch_size, const I* batch_row_ind, const I* batch_col_ind,
															 const T* A, T beta) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= batch_size) {
		return;
	}

	I row = batch_row_ind[idx];
	I col = batch_col_ind[idx];
	I row_start = csr_row_ptr[row];
	I row_end = csr_row_ptr[row + 1];
	for(I k = row_start; k < row_end; k++) {
		if(csr_col_ind[k] == col) {
			matval[k] = beta * A[idx] + alpha * matval[k];
		}
	}
}


template <typename I, typename T>
__global__ void
MatrixCSRSetValueBlockedBatchedKernel(T* matval, T alpha,
																			I csr_num_row, I csr_num_col,
																	 		const I* csr_row_ptr, const I* csr_col_ind,
																	 		I batch_size, const I* batch_row_ind, const I* batch_col_ind,
																	 		I block_row, I block_col,
																	 		const T* A, T beta, int lda, int stride) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= batch_size) {
		return;
	}

	I row = batch_row_ind[idx];
	I col = batch_col_ind[idx];
	I row_start = csr_row_ptr[row], row_end = csr_row_ptr[row + 1];
	I len = row_end - row_start;
	I k;
	for(k = row_start; k < row_end; k++) {
		if(csr_col_ind[k] == col) {
			break;
		}
	}
	matval += k * block_row * block_col;
	A += idx * stride;
	for(I ii = 0; ii < block_row; ii++) {
		for(I jj = 0; jj < block_col; jj++) {
			matval[ii * len * block_col + jj]
				= beta * A[idx * stride + ii * lda + jj] 
				+ alpha * matval[ii * block_col + jj];
		}
	}
}

template <typename I, typename T>
__global__ void
SetBlockValueToSubmatKernel(T* matval[], T alpha, I n_offset, const I* offset, I nshl,
													  I batch_size, const I* batch_index_ptr, const I* ien,
													  I num_row, I num_col, const I* row_ptr, const I* col_ind,
													  const T* val, int lda, int stride, T beta, const I* mask) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	I iel, row, col, aa, bb;

	I block_row, block_col;

	I k, start, end, len;

	T* m = NULL, a, b;
	b32 found = FALSE;

	if(idx >= batch_size * nshl * nshl) {
		return;
	}
	if(mask && mask[idx / (nshl * nshl)] == 0) {
		return;
	}

	iel = batch_index_ptr[idx / (nshl * nshl)];
	aa = idx % (nshl * nshl) / nshl;
	bb = idx % nshl;


	row = ien[iel * nshl + aa];
	col = ien[iel * nshl + bb];
	found = row == 0 && col == 0;
	found = FALSE;

	start = row_ptr[row];
	end = row_ptr[row + 1];
	len = end - start;

	for(k = start; k < end; k++) {
		if(col_ind[k] == col) {
			break;
		}
	}

	val += idx * stride;

	if(found) {
		printf("iel=%d, row=%d, col=%d, k=%d, len=%d\n", iel, row, col, k, len);
	}

	for(I i = 0; i < n_offset; i++) {
		block_row = offset[i + 1] - offset[i];
		for(I j = 0; j < n_offset; j++) {
			block_col = offset[j + 1] - offset[j];
			m = matval[i * n_offset + j];
			if(m == NULL) {
				continue;
			}
			found = FALSE;
			// if(i == 0 && j == 1) {
			// 	found = TRUE;
			// }
			m += start * block_row * block_col + (k - start) * block_col;
			for(I ii = 0; ii < block_row; ii++) {
				for(I jj = 0; jj < block_col; jj++) {
					if(start * block_row * block_col + (k - start) * block_col + ii * len * block_col + jj >= row_ptr[num_row] * block_row * block_col) {
						printf("start=%d, k=%d, ii=%d, jj=%d, len=%d, row_ptr[%d]=%d\n", start, k, ii, jj, len, num_row, row_ptr[num_row]);
					}
					a = m[ii * len * block_col + jj];
					b = val[(offset[i] + ii) * lda + (offset[j] + jj)];
					if(found && b != 0.0) {
						printf("(m[%d][%d])[%d*%d*%d+%d*%d*%d+%d] = %g \n",
									 i, j,
									 k, block_row, block_col,
									 ii, len, block_col, jj,
									 a);
						printf("val[%d*%d+%d] = %g\n", offset[i] + ii, lda, offset[j] + jj, b);
					}
					m[ii * len * block_col + jj] = alpha * a + beta * b;
					// atomicAdd(&m[ii * len * block_col + jj], b);
				}
			}
		}
	}
}

template <typename I, typename T>
__global__ void
SetValKernel(T* val, I n, T alpha) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= n) {
		return;
	}
	val[i] = alpha;
}

__BEGIN_DECLS__

void SetValGPU(value_type* val, index_type n, value_type alpha) {
	dim3 block(256);
	dim3 grid((n + block.x - 1) / block.x);
	SetValKernel<<<grid, block>>>(val, n, alpha);
}
void MatrixCSRZeroRowGPU(value_type* matval,
												 index_type num_row, index_type num_col, const index_type* row_ptr, const index_type* col_ind,
												 index_type n, const index_type* row, index_type shift, value_type diag) {
	size_t block_dim = 256;
	size_t grid_dim = (num_row + block_dim - 1) / block_dim;
	MatrixCSRZeroRowKernel<index_type, value_type><<<grid_dim, block_dim>>>(matval,
																																					num_row, num_col, row_ptr, col_ind,
																																					n, row, shift, diag);
}

void MatrixCSRGetDiagGPU(const value_type* val, const index_type* row_ptr, const index_type* col_idx, value_type* diag, index_type num_row) {
	size_t block_dim = 256;
	size_t grid_dim = (num_row + block_dim - 1) / block_dim;
	MatrixCSRGetDiagKernel<value_type, index_type><<<grid_dim, block_dim>>>(val, row_ptr, col_idx, diag, num_row);
}

void MatrixCSRSetValuesCOOGPU(value_type* matval, value_type alpha,
															index_type num_row, index_type num_col,
															const index_type* row_ptr, const index_type* col_ind,
															index_type n, const index_type* row, const index_type* col,
															const value_type* val, value_type beta) {
	size_t block_dim = 256;
	size_t grid_dim = (n + block_dim - 1) / block_dim;
	MatrixCSRSetValuesCOOKernel<<<grid_dim, block_dim>>>(matval, alpha,
																											 num_row, num_col,
																											 row_ptr, col_ind,
																											 n, row, col,
																											 val, beta);
}

void MatrixCSRSetValuesIndGPU(value_type* matval, value_type alpha,
															index_type n, const index_type* ind,
															const value_type* val, value_type beta) {
	size_t block_dim = 256;
	size_t grid_dim = (n + block_dim - 1) / block_dim;
	MatrixCSRSetValuesIndKernel<<<grid_dim, block_dim>>>(matval, alpha,
																											 n, ind, val, beta);
}

void MatrixCSRAddElementLHSGPU(value_type* matval, index_type nshl, index_type bs, 
															 index_type num_row, const index_type* row_ptr,
															 index_type num_col, const index_type* col_ind,
															 index_type batch_size, const index_type* batch_ptr, const index_type* ien,
															 const value_type* val, int lda) {
	size_t block_dim = 256 * 4;
	size_t grid_dim = (batch_size + block_dim - 1) / block_dim;

	MatrixCSRAddElementLHSKernel<index_type, value_type><<<grid_dim, block_dim>>>(matval, nshl, bs,
																																								num_row, row_ptr,
																																								num_col, col_ind,
																																								batch_size, batch_ptr, ien,
																																								val, lda);
}

void MatrixFSElementLHSGPU(Matrix* mat,
															 index_type nshl, index_type bs,
															 index_type num_row, const index_type* row_ptr,
															 index_type num_col, const index_type* col_ind,
															 index_type batch_size, const index_type* batch_ptr, const index_type* ien,
															 const value_type* val, int lda) {
	const size_t block_dim = 256 * 4;
	const size_t grid_dim = (nshl * nshl * batch_size + block_dim - 1) / block_dim;
	MatrixFS* mat_nested = (MatrixFS*)mat->data;

#define Kernel MatrixFSAddElementLHSKernel<Matrix, value_type, index_type>
	Kernel<<<grid_dim, block_dim>>>(mat_nested->mat, mat_nested->n_offset, mat_nested->d_offset,
																	nshl, num_row, row_ptr, num_col, col_ind,
																	batch_size, batch_ptr, ien, val, lda);
#undef Kernel
}

void MatrixCSRSetValueBatchedGPU(value_type* matval, value_type alpha,
																 index_type csr_num_row, index_type csr_num_col,
																 const index_type* csr_row_ptr, const index_type* csr_col_ind,
																 index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
																 const value_type* A, value_type beta) {
	const size_t block_dim = 256;
	const size_t grid_dim = (batch_size + block_dim - 1) / block_dim;
	MatrixCSRSetValueBatchedKernel<index_type, value_type><<<grid_dim, block_dim>>>(matval, alpha,
																																									csr_num_row, csr_num_col,
																																									csr_row_ptr, csr_col_ind,
																																									batch_size, batch_row_ind, batch_col_ind,
																																									A, beta);
}


void MatrixCSRAddElemValueBatchedGPU(value_type* matval, value_type alpha,
																		 index_type batch_size, const index_type* batch_index_ptr, const index_type* ien, index_type nshl,
																		 index_type num_row, index_type num_col, const index_type* row_ptr, const index_type* col_ind,
																		 const value_type* val, value_type beta, const index_type* mask) {
	size_t block_dim = 256;
	size_t grid_dim = (batch_size * nshl * nshl + block_dim - 1) / block_dim;

	if(mask == NULL) {
		MatrixCSRAddElemValueBatchedKernel<<<grid_dim, block_dim>>>(matval, alpha,
																																batch_size, batch_index_ptr, ien, nshl,
																																num_row, num_col, row_ptr, col_ind,
																																val, beta);
	}
	else {
		MatrixCSRAddElemValueBatchedMaskedKernel<<<grid_dim, block_dim>>>(matval, alpha,
																																		batch_size, batch_index_ptr, ien, nshl,
																																		num_row, num_col, row_ptr, col_ind,
																																		val, beta, mask);
	}
}

void MatrixCSRAddElemValueBlockedBatchedGPU(value_type* matval, value_type alpha,
																					index_type batch_size, const index_type* batch_index_ptr, const index_type* ien, index_type nshl,
																					index_type num_row, index_type num_col, const index_type* row_ptr, const index_type* col_ind,
																					index_type block_row, index_type block_col,
																					const value_type* val, int lda, int stride, value_type beta, const index_type* mask) {
	size_t block_dim = 256;
	size_t grid_dim = (batch_size * nshl * nshl + block_dim - 1) / block_dim;

	if(mask == NULL) {
		MatrixCSRAddElemValueBlockedBatchedKernel<<<grid_dim, block_dim>>>(matval, alpha,
																																			batch_size, batch_index_ptr, ien, nshl,
																																			num_row, num_col, row_ptr, col_ind,
																																			block_row, block_col,
																																			val, lda, stride, beta);
	}
	else {
		MatrixCSRAddElemValueBlockedBatchedMaskedKernel<<<grid_dim, block_dim>>>(matval, alpha,
																																					batch_size, batch_index_ptr, ien, nshl,
																																					num_row, num_col, row_ptr, col_ind,
																																					block_row, block_col,
																																					val, lda, stride, beta, mask);
	}
}

void MatrixCSRSetValueBlockedBatchedGPU(value_type* matval, value_type alpha,
																				index_type batch_size, const index_type* batch_index_ptr, const index_type* ien, index_type nshl,
																				index_type num_row, index_type num_col, const index_type* row_ptr, const index_type* col_ind,
																				index_type block_row, index_type block_col,
																				const value_type* val, int lda, int stride, value_type beta) {
	size_t block_dim = 256;
	size_t grid_dim = (batch_size * nshl * nshl + block_dim - 1) / block_dim;

#define Kernel MatrixCSRSetValueBlockedBatchedKernel<index_type, value_type>
	Kernel<<<grid_dim, block_dim>>>(matval, alpha,
																 num_row, num_col, row_ptr, col_ind,
																 batch_size, batch_index_ptr, ien,
																 block_row, block_col,
																 val, beta, lda, stride);
#undef Kernel
	// MatrixCSRAddElemValueBlockedBatchedKernel<<<grid_dim, block_dim>>>(matval, alpha,
	// 																																	 batch_size, batch_index_ptr, ien, nshl,
	// 																																	 num_row, num_col, row_ptr, col_ind,
	// 																																	 block_row, block_col,
	// 																																	 val, lda, stride, beta);

}

void SetBlockValueToSubmatGPU(value_type** matval, value_type alpha,
															index_type n_offset, const index_type* offset, index_type nshl,
															index_type batch_size, const index_type* batch_index_ptr, const index_type* ien,
															index_type num_row, index_type num_col,
															const index_type* row_ptr, const index_type* col_ind,
															const value_type* val, int lda, int stride, value_type beta, const index_type* mask) {
	size_t block_dim = 256;
	size_t grid_dim = CEIL_DIV(batch_size * nshl * nshl, block_dim);
	SetBlockValueToSubmatKernel<<<grid_dim, block_dim>>>(matval, alpha,
																											 n_offset, offset, nshl,
																											 batch_size, batch_index_ptr, ien,
																											 num_row, num_col,
																											 row_ptr, col_ind,
																											 val, lda, stride, beta, mask);
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

void MatrixGetDiagBlockGPU(const value_type* matval,
													 index_type block_size,
													 index_type num_row, index_type num_col,
													 const index_type* row_ptr, const index_type* col_idx,
													 value_type* diag_block, int lda, int stride) {
	int num_thread = 256;
	int num_block = (num_row + num_thread - 1) / num_thread;
	MatrixGetDiagBlockKernel<<<num_block, num_thread>>>(matval, block_size, num_row, num_col, row_ptr, col_idx, diag_block, lda, stride);
}


__END_DECLS__
#endif
