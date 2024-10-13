#include "alloc.h"
#include "blas.h"
#include "sequential_matrix.h"
#include "sequential_matrix_device.h"

static void MatMultAddDenseDensePrivate(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	index_type m, n, k;
	value_type* dataA = SeqMatAsType(A, SeqMatDense)->data;
	value_type* dataB = SeqMatAsType(B, SeqMatDense)->data;
	
	m = SeqMatNumRow(A);
	n = SeqMatNumCol(B);
	k = SeqMatNumCol(A);
	ASSERT(k == SeqMatNumRow(B) && "Matrix A must have the same number of columns as matrix B.");


	if(reuse == MAT_INITIAL) {
		CdamFree(SeqMatAsType(C, SeqMatDense)->data,
													sizeof(value_type) * SeqMatNumRow(C) * SeqMatNumCol(C),
													DEVICE_MEM);
		SeqMatAsType(C, SeqMatDense)->data = CdamTMalloc(value_type, m * n, DEVICE_MEM);
		CdamMemset(SeqMatAsType(C, SeqMatDense)->data, 0, m * n * sizeof(value_type), DEVICE_MEM);
		SeqMatNumRow(C) = m;
		SeqMatNumCol(C) = n;
	}
	else {
		ASSERT(m == SeqMatNumRow(C) && "Matrix C must have the same number of rows as matrix A.");
		ASSERT(n == SeqMatNumCol(C) && "Matrix C must have the same number of columns as matrix B.");
	}
	dgemm(BLAS_N, BLAS_N, n, m, k, alpha, dataB, n, dataA, k, beta, SeqMatAsType(C, SeqMatDense)->data, n);

}
static void MatMultAddDenseCSRPrivate(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);
static void MatMultAddCSRCSRPrivate(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);
static void MatMultAddCSRDensePrivate(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	ASSERT(0 && "Not implemented yet.");
}

static void SeqMatSetupDensePrivate(void* A) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);

	SeqMatAsType(A, SeqMatDense)->data = CdamTMalloc(value_type, nrow * ncol, DEVICE_MEM);
}

static void SeqMatDestroyDensePrivate(void* A) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	CdamFree(SeqMatAsType(A, SeqMatDense)->data, sizeof(value_type) * nrow * ncol, DEVICE_MEM);
	CdamFree(SeqMatAsType(A, SeqMatDense), sizeof(SeqMatDense), HOST_MEM);
}

static void SeqMatZeroDensePrivate(void* A) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	value_type* data = SeqMatAsType(A, SeqMatDense)->data;
	CdamMemset(data, 0, nrow * ncol * sizeof(value_type), DEVICE_MEM);
}

static void SeqMatZereRowDensePrivate(void* A, index_type row, index_type* rows, index_type shift, value_type diag) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	value_type* data = SeqMatAsType(A, SeqMatDense)->data;
#ifdef CDAM_USE_CUDA
	SeqMatZeroRowDenseGPU(data, nrow, ncol, row, rows, shift, 0, diag, 0);
#else
	index_type i, ir;
	for(i = 0; i < row; i++) {
		ir = rows[i] + shift;
		if(ir >= 0 && ir < nrow) {
			CdamMemset(data + ir * ncol, 0, ncol * sizeof(value_type), DEVICE_MEM);
			if(ir < ncol) {
				CdamMemcpy(data + ir * ncol + ir, &diag, sizeof(value_type), DEVICE_MEM, HOST_MEM);
			}
		}
	}
#endif /* CDAM_USE_CUDA */
}

static void SeqMatGetSubmatDensePrivate(void* A, index_type nr, index_type* row,
																				index_type nc, index_type* col, void* B) {
	ASSERT(SeqMatType(A) == MAT_TYPE_DENSE && "Matrix A must be dense.");
	ASSERT(SeqMatType(B) == MAT_TYPE_VIRTUAL && "Matrix B must be virtual.");

	SeqMatNumRow(B) = nr;
	SeqMatNumCol(B) = nc;

	if(SeqMatData(B) == NULL) {
		SeqMatData(B) = CdamTMalloc(SeqMatVirtual, 1, HOST_MEM);
		CdamMemset(SeqMatAsType(B, SeqMatVirtual), 0, sizeof(SeqMatVirtual), HOST_MEM);
	}
	SeqMatAsType(B, SeqMatVirtual)->parent = (SeqMat*)A;
	SeqMatAsType(B, SeqMatVirtual)->row = CdamTMalloc(index_type, nr, DEVICE_MEM);
	CdamMemcpy(SeqMatAsType(B, SeqMatVirtual)->row, row, nr * sizeof(index_type), DEVICE_MEM, DEVICE_MEM);
	SeqMatAsType(B, SeqMatVirtual)->col = CdamTMalloc(index_type, nc, DEVICE_MEM);
	CdamMemcpy(SeqMatAsType(B, SeqMatVirtual)->col, col, nc * sizeof(index_type), DEVICE_MEM, DEVICE_MEM);
}

static void SeqMatTransposeDensePrivate(void* A) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	value_type* data = SeqMatAsType(A, SeqMatDense)->data;
	SeqMatNumRow(A) = ncol;
	SeqMatNumCol(A) = nrow;
	dtranspose(nrow, ncol, data, ncol, data, nrow);
}

static void SeqMatMultAddDensePrivate(value_type alpha, void* A, value_type *x, value_type beta, value_type *y) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	value_type* data = SeqMatAsType(A, SeqMatDense)->data;

	dgemv(BLAS_T, ncol, nrow, alpha, data, ncol, x, 1, beta, y, 1);
}

static void SeqMatMultTransposeAddDensePrivate(value_type alpha, void* A, value_type *x, value_type beta, value_type *y) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	value_type* data = SeqMatAsType(A, SeqMatDense)->data;

	dgemv(BLAS_N, ncol, nrow, alpha, data, ncol, x, 1, beta, y, 1);
}

static void SeqMatMatMultAddDensePrivate(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	MatType typeB = SeqMatType(B);

	if(typeB == MAT_TYPE_DENSE) {
		MatMultAddDenseDensePrivate(alpha, A, B, beta, C, reuse);
	}
	else if(typeB == MAT_TYPE_CSR) {
		/* AB= (AB)^T^T = (B^T A^T)^T */
		SeqMatTranspose(A);
		SeqMatTranspose(B);
		MatMultAddCSRDensePrivate(alpha, B, A, beta, C, reuse);
		SeqMatTranspose(A);
		SeqMatTranspose(B);
		SeqMatTranspose(C);
	}
	else {
		ASSERT(0 && "Unsupported matrix type.");
	}
}

static void SeqMatMatMultTransposeAddDensePrivate(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	/* A^T B = (B^T A)^T */
	SeqMatTranspose(A);
	SeqMatMatMultAddDensePrivate(alpha, A, B, beta, C, reuse);
	SeqMatTranspose(A);
}

static void SeqMatGetDiagDensePrivate(void* A, value_type* diag, index_type bs) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);

	ASSERT(nrow == ncol && "Matrix must be square.");
	ASSERT(nrow % bs == 0 && "Block size must be a divisor of the matrix size.");
#ifdef CDAM_USE_CUDA
	SeqMatGetDiagDenseGPU(SeqMatAsType(A, SeqMatDense)->data, nrow, diag, bs, 0);
#else
	index_type i;
	value_type* data = SeqMatAsType(A, SeqMatDense)->data;
	for(i = 0; i < nrow * bs; ++i) {
		diag[i] = data[(i / bs) * nrow + i / (bs * bs) * bs + i % bs];
	}
#endif
}


static void SeqMatAddElemValueBatchedDensePrivate(void* A, index_type nelem, index_type nshl, index_type *ien,
																									byte* row_mask, byte* col_mask,
																									value_type* value, Arena scratch) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	value_type* data = SeqMatAsType(A, SeqMatDense)->data;
	/* Logically, shape(value) = (batch_size, nshl, nshl, block_row, block_col */
	/* Physically, shape(value) = (batch_size, nshl, nshl, stride) */
	/* stride >= block_row * ldv && ldv > block_col */
	/* Each block in value is stored in a row-major order. */

#ifdef CDAM_USE_CUDA
	CdamLayout* d_rmap = SeqMatRowLayoutDevice(A);
	CdamLayout* d_cmap = SeqMatColLayoutDevice(A);
	SeqMatAddElemValueBatchedDenseGPU(data, nrow, ncol,
																		d_rmap, d_cmap,
																		row_mask, col_mask,
																		nelem, nshl, ien,
																		value, scratch, 0);
#else
	CdamLayout* rmap = SeqMatRowLayout(A);
	CdamLayout* cmap = SeqMatColLayout(A);
	index_type block_row = CdamLayoutComponentOffset(rmap)[CdamLayoutNumComponent(rmap)];
	index_type block_col = CdamLayoutComponentOffset(cmap)[CdamLayoutNumComponent(cmap)];
	index_type r, c, i;
	index_type ishl, jshl, ishg, jshg;
	index_type dst_row, dst_col;
	value_type* value_ptr;
	for(i = 0; i < nelem; i++) {
		for(ishl = 0; ishl < nshl; ishl++) {
			if((row_mask[i] & (1 << ishl)) == 0) continue;
			ishg = ien[i * nshl + ishl];
			for(jshl = 0; jshl < nshl; jshl++) {
				if((col_mask[i] & (1 << jshl)) == 0) continue;
				jshg = ien[i * nshl + jshl];
				value_ptr = value + i * nshl * nshl * block_row * block_col + (ishl * nshl + jshl) * block_row * block_col;
				
				for(r = 0; r < block_row; ++r) {
					dst_row = CdamLayoutDOFMapLocal(rmap, ishg, r);
					for(c = 0; c < block_col; ++c) {
						dst_col = CdamLayoutDOFMapLocal(cmap, jshg, c);
						data[dst_row * ncol + dst_col] += value_ptr[r * block_col + c];
					}
				}
			}
		}
	}
#endif
}

static void SeqMatSetupCSRPrivate(void* A) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);

	CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;

	SeqMatAsType(A, SeqMatCSR)->data = CdamTMalloc(value_type, spy->nnz, DEVICE_MEM);
	CdamMemset(SeqMatAsType(A, SeqMatCSR)->data, 0, spy->nnz * sizeof(value_type), DEVICE_MEM);

	SpMatCreate(&SeqMatAsType(A, SeqMatCSR)->descr, nrow, ncol, spy->nnz, CSRAttrRowPtr(spy), CSRAttrColInd(spy), SeqMatAsType(A, SeqMatCSR)->data);

	value_type* x = CdamTMalloc(value_type, ncol, DEVICE_MEM);
	value_type* y = CdamTMalloc(value_type, nrow, DEVICE_MEM);

	dspmvBufferSize(SP_N, 1.0, SeqMatAsType(A, SeqMatCSR)->descr, x, 0.0, y, &SeqMatAsType(A, SeqMatCSR)->buffer_size);
	SeqMatAsType(A, SeqMatCSR)->buffer = CdamTMalloc(char, SeqMatAsType(A, SeqMatCSR)->buffer_size, DEVICE_MEM);

	dspmvPreprocess(SP_N, 1.0, SeqMatAsType(A, SeqMatCSR)->descr, x, 0.0, y, SeqMatAsType(A, SeqMatCSR)->buffer);

	CdamFree(x, sizeof(value_type) * ncol, DEVICE_MEM);
	CdamFree(y, sizeof(value_type) * nrow, DEVICE_MEM);

}

static void SeqMatDestroyCSRPrivate(void* A) {
	CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;
	CdamFree(SeqMatAsType(A, SeqMatCSR)->data, sizeof(value_type) * spy->nnz, DEVICE_MEM);
	SpMatDestroy(SeqMatAsType(A, SeqMatCSR)->descr);
	CdamFree(SeqMatAsType(A, SeqMatCSR)->buffer, SeqMatAsType(A, SeqMatCSR)->buffer_size, DEVICE_MEM);
	CdamFree(SeqMatAsType(A, SeqMatCSR), sizeof(SeqMatCSR), HOST_MEM);
}

static void SeqMatZeroCSRPrivate(void* A) {
	value_type* data = SeqMatAsType(A, SeqMatCSR)->data;
	CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;
	CdamMemset(data, 0, spy->nnz * sizeof(value_type), DEVICE_MEM);
}

static void SeqMatZeroRowCSRPrivate(void* A, index_type row, index_type* rows, index_type shift, value_type diag) {
	CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;
	value_type* data = SeqMatAsType(A, SeqMatCSR)->data;

#ifdef CDAM_USE_CUDA
	SeqMatZeroRowCSRGPU(data, spy, row, rows, shift, diag, 0);
#else
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	index_type* row_ptr = CSRAttrRowPtr(spy);
	index_type* col_ind = CSRAttrColInd(spy);
	index_type i, j, ir;
	for(i = 0; i < row; i++) {
		ir = rows[i] + shift;
		if(ir >= 0 && ir < nrow) {
			CdamMemset(data + col_ind[row_ptr[ir]], 0, (row_ptr[ir + 1] - row_ptr[ir]) * sizeof(value_type), DEVICE_MEM);
			for(j = row_ptr[ir]; ir < ncol && j < row_ptr[ir + 1]; j++) {
				if(col_ind[j] == ir) {
					data[j] = diag;
					break;
				}
			}
		}
	}
#endif /* CDAM_USE_CUDA */
}

static void SeqMatGetSubmatCSRPrivate(void* A, index_type nr, index_type* row,
																			index_type nc, index_type* col, void* B) {
	// index_type i, j;
	// index_type nrow = SeqMatNumRow(A);
	// index_type ncol = SeqMatNumCol(A);
	// CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;
	// value_type* dataA = SeqMatAsType(A, SeqMatCSR)->data;

	ASSERT(SeqMatType(A) == MAT_TYPE_CSR && "Matrix A must be CSR.");
	ASSERT(SeqMatType(B) == MAT_TYPE_VIRTUAL && "Matrix B must be virtual.");

	if(SeqMatData(B) == NULL) {
		SeqMatData(B) = CdamTMalloc(SeqMatVirtual, 1, HOST_MEM);
		CdamMemset(SeqMatAsType(B, SeqMatVirtual), 0, sizeof(SeqMatVirtual), HOST_MEM);
	}
	SeqMatNumRow(B) = nr;
	SeqMatNumCol(B) = nc;
	SeqMatAsType(B, SeqMatVirtual)->parent = (SeqMat*)A;
	SeqMatAsType(B, SeqMatVirtual)->row = CdamTMalloc(index_type, nr, DEVICE_MEM);
	CdamMemcpy(SeqMatAsType(B, SeqMatVirtual)->row, row, nr * sizeof(index_type), DEVICE_MEM, DEVICE_MEM);
	SeqMatAsType(B, SeqMatVirtual)->col = CdamTMalloc(index_type, nc, DEVICE_MEM);
	CdamMemcpy(SeqMatAsType(B, SeqMatVirtual)->col, col, nc * sizeof(index_type), DEVICE_MEM, DEVICE_MEM);
}

static void SeqMatTransposeCSRPrivate(void* A) {
	// index_type nrow = SeqMatNumRow(A);
	// index_type ncol = SeqMatNumCol(A);
	// CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;

	ASSERT(A && "Matrix A is NULL.");

	ASSERT(0 && "Not implemented yet.");
}

static void SeqMatMultAddCSRPrivate(value_type alpha, void* A, value_type* x, value_type beta, value_type* y) {
	dspmv(SP_N, alpha, SeqMatAsType(A, SeqMatCSR)->descr, x, beta, y, SeqMatAsType(A, SeqMatCSR)->buffer);
}

static void SeqMatMultTransposeAddCSRPrivate(value_type alpha, void* A, value_type* x, value_type beta, value_type* y) {
	dspmv(SP_T, alpha, SeqMatAsType(A, SeqMatCSR)->descr, x, beta, y,SeqMatAsType(A, SeqMatCSR)->buffer);
}

static void SeqMatMatMultAddCSRPrivate(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	/* TODO: Wrap those functions in blas.h */
	/* CSR * CSR => SpGEMM or SpGEMMreuse */
	/* CSR * Dense => SpMM */
	ASSERT(0 && "Not implemented yet.");
}
static void SeqMatMatMultTransposeAddCSRPrivate(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	/* A^T * B = (B^T * A)^T */
	SeqMatTranspose(A);
	SeqMatMatMultAddCSRPrivate(alpha, A, B, beta, C, reuse);
	SeqMatTranspose(A);
}

static void SeqMatGetDiagCSRPrivate(void* A, value_type* diag, index_type bs) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);

	ASSERT(nrow == ncol && "Matrix must be square.");

	if(bs >= 1) {
#ifdef CDAM_USE_CUDA
		SeqMatGetDiagBlockedCSRGPU(SeqMatAsType(A, SeqMatCSR)->data, SeqMatAsType(A, SeqMatCSR)->spy, diag, bs, bs * bs, bs, 0);
#else
		index_type i, j, ir;
		index_type ib, lane;
		value_type* data = SeqMatAsType(A, SeqMatCSR)->data;
		CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;
		for(i = 0; i < nrow; i++) {
			ib = i / bs;
			lane = i % bs;
			for(j = spy->row_ptr[i]; j < spy->row_ptr[i + 1]; j++) {
				ir = spy->col_ind[j];
				if(ir >= ib * bs && ir < (ib + 1) * bs) {
					diag[ib * bs * bs + lane * bs + ir - ib * bs] = data[j];
				}
			}
		}
#endif
	}
	else {
		ASSERT(0 && "Block size must be greater than 0.");
	}
}

static void SeqMatAddElemValueBatchedCSRPrivate(void* A,
																								index_type nelem, index_type nshl,
																								index_type *ien,
																								byte* row_mask, byte* col_mask,
																								value_type* value, Arena scratch) {
	CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;
	value_type* data = SeqMatAsType(A, SeqMatCSR)->data;


#ifdef CDAM_USE_CUDA
	CdamLayout* d_rmap = SeqMatRowLayoutDevice(A);
	CdamLayout* d_cmap = SeqMatColLayoutDevice(A);
	SeqMatAddElemValueBatchedCSRGPU(data, spy,
																	d_rmap, d_cmap,
																	row_mask, col_mask,
																	nelem, nshl, ien,
																	value, scratch, 0);
#else
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);

	CdamLayout* rmap = SeqMatRowLayout(A);
	CdamLayout* cmap = SeqMatColLayout(A);
	index_type r, c, k;
	index_type ishl, jshl, ishg, jshg;
	index_type dst_row, dst_col;
	index_type i, iel;
	index_type block_row = CdamLayoutComponentOffset(rmap)[CdamLayoutNumComponent(rmap)];
	index_type block_col = CdamLayoutComponentOffset(cmap)[CdamLayoutNumComponent(cmap)];
	index_type start, end, len;

	for(i = 0; i < nelem * nshl * nshl; ++i) {
		iel = i / (nshl * nshl);
		ishl = (i % (nshl * nshl)) / nshl; 
		jshl = i % nshl;

		if((row_mask[iel] & (1 << ishl)) == 0) continue;
		if((col_mask[iel] & (1 << jshl)) == 0) continue;

		ishg = ien[iel * nshl + ishl];
		jshg = ien[iel * nshl + jshl];

		for(r = 0; r < block_row; ++r) {
			dst_row = CdamLayoutDOFMapLocal(rmap, ishg, r);
			start = spy->row_ptr[dst_row];
			end = spy->row_ptr[dst_row + 1];
			len = end - start;
			k = start;
			for(c = 0; c < block_col; ++c) {
				dst_col = CdamLayoutDOFMapLocal(cmap, jshg, c);
				for(; k < end; ++k) {
					if(spy->col_ind[k] == dst_col) {
						data[k] += value[i];
						break;
					}
				}
			}
		}
	}
#endif
}

static void SeqMatSetupNestedPrivate(void* A) {
	index_type nrow = SeqMatNumRow(A);
	// index_type ncol = SeqMatNumCol(A);
	SeqMatData(A) = CdamTMalloc(void*, nrow, HOST_MEM);
	CdamMemset(SeqMatData(A), 0, nrow * sizeof(void*), HOST_MEM);
}


void SeqMatCreate(MatType type, index_type nrow, index_type ncol, void** A) {
	*A = CdamTMalloc(SeqMat, 1, HOST_MEM);
	SeqMat* mat = *A;
	CdamMemset(*A, 0, sizeof(SeqMat), HOST_MEM);
	SeqMatType(*A) = type;
	SeqMatNumRow(*A) = nrow;
	SeqMatNumCol(*A) = ncol;
	if(type == MAT_TYPE_DENSE) {
		SeqMatData(*A) = CdamTMalloc(SeqMatDense, 1, HOST_MEM);
		CdamMemset(SeqMatAsType(*A, SeqMatDense), 0, sizeof(SeqMatDense), HOST_MEM);
		mat->op->setup = SeqMatSetupDensePrivate;
		mat->op->destroy = SeqMatDestroyDensePrivate;
		mat->op->zero = SeqMatZeroDensePrivate;
		mat->op->zero_row = SeqMatZereRowDensePrivate;
		mat->op->get_submat = NULL;
		// mat->op->copy = SeqMatCopyDensePrivate;
		mat->op->transpose = SeqMatTranspose;
		mat->op->multadd = SeqMatMultAddDensePrivate;
		mat->op->multtransposeadd = SeqMatMultTransposeAddDensePrivate;
		mat->op->matmultadd = SeqMatMatMultAddDensePrivate;
		mat->op->mattransposemultadd = SeqMatMatMultTransposeAddDensePrivate;
		mat->op->get_diag = SeqMatGetDiagDensePrivate;
		mat->op->add_elem_value_batched = SeqMatAddElemValueBatchedDensePrivate;
	}
	else if(type == MAT_TYPE_CSR) {
		SeqMatData(*A) = CdamTMalloc(SeqMatCSR, 1, HOST_MEM);
		CdamMemset(SeqMatAsType(*A, SeqMatCSR), 0, sizeof(SeqMatCSR), HOST_MEM);
		mat->op->setup = SeqMatSetupCSRPrivate;
		mat->op->destroy = SeqMatDestroyCSRPrivate;
		mat->op->zero = SeqMatZeroCSRPrivate;
		mat->op->zero_row = SeqMatZeroRowCSRPrivate;
		mat->op->get_submat = NULL;
		// mat->op->copy = SeqMatCopyCSRPrivate;
		mat->op->transpose = SeqMatTransposeCSRPrivate;
		mat->op->multadd = SeqMatMultAddCSRPrivate;
		mat->op->multtransposeadd = SeqMatMultTransposeAddCSRPrivate;
		mat->op->matmultadd = SeqMatMatMultAddCSRPrivate;
		mat->op->mattransposemultadd = SeqMatMatMultTransposeAddCSRPrivate;
		mat->op->get_diag = SeqMatGetDiagCSRPrivate;
		mat->op->add_elem_value_batched = SeqMatAddElemValueBatchedCSRPrivate;
	}
	else {
		ASSERT(0 && "Unsupported matrix type.");
	}
}

void SeqMatSetup(void* A) {
	SeqMat* mat = (SeqMat*)A;
	if(mat)
		mat->op->setup(A);
}
void SeqMatDestroy(void* A) {
	SeqMat* mat = (SeqMat*)A;
	if(mat)
		mat->op->destroy(A);
	CdamFree(A, sizeof(SeqMat), HOST_MEM);
}

void SeqMatZero(void* A) {
	SeqMat* mat = (SeqMat*)A;
	if(mat)
		mat->op->zero(A);
}

void SeqMatZeroRow(void* A, index_type row, index_type* rows, index_type shift, value_type diag) {
	SeqMat* mat = (SeqMat*)A;
	if(mat)
		mat->op->zero_row(A, row, rows, shift, diag);
}

void SeqMatGetSubmat(void* A, index_type nr, index_type* row, index_type nc, index_type* col, void* B) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);

	ASSERT(SeqMatType(B) == MAT_TYPE_VIRTUAL && "Matrix B must be virtual.");

	SeqMatNumRow(B) = nr;
	SeqMatNumCol(B) = nc;

	if(SeqMatData(B) == NULL) {
		SeqMatData(B) = CdamTMalloc(SeqMatVirtual, 1, HOST_MEM);
		CdamMemset(SeqMatAsType(B, SeqMatVirtual), 0, sizeof(SeqMatVirtual), HOST_MEM);
	}
	SeqMatAsType(B, SeqMatVirtual)->parent = (SeqMat*)A;
	SeqMatAsType(B, SeqMatVirtual)->row = CdamTMalloc(index_type, nr, DEVICE_MEM);
	CdamMemcpy(SeqMatAsType(B, SeqMatVirtual)->row, row, nr * sizeof(index_type), DEVICE_MEM, DEVICE_MEM);
	SeqMatAsType(B, SeqMatVirtual)->col = CdamTMalloc(index_type, nc, DEVICE_MEM);
	CdamMemcpy(SeqMatAsType(B, SeqMatVirtual)->col, col, nc * sizeof(index_type), DEVICE_MEM, DEVICE_MEM);

	SeqMatAsType(B, SeqMatVirtual)->prolonged_input = CdamTMalloc(value_type, ncol, DEVICE_MEM);
	SeqMatAsType(B, SeqMatVirtual)->prolonged_output = CdamTMalloc(value_type, nrow, DEVICE_MEM);
}

// void SeqMatCopy(void* A, void* B) {
// 	SeqMat* mat = (SeqMat*)A;
// 	if(mat)
// 		mat->op->copy(A, B);
// }

void SeqMatTranspose(void* A) {
	SeqMat* mat = (SeqMat*)A;
	if(mat)
		mat->op->transpose(A);
}

void SeqMatMultAdd(value_type alpha, void* A, value_type* x, value_type beta, value_type* y) {
	SeqMat* mat = (SeqMat*)A;
	if(mat)
		mat->op->multadd(alpha, A, x, beta, y);
}

void SeqMatMultTransposeAdd(value_type alpha, void* A, value_type* x, value_type beta, value_type* y) {
	SeqMat* mat = (SeqMat*)A;
	if(mat)
		mat->op->multtransposeadd(alpha, A, x, beta, y);
}
void SeqMatMatMultAdd(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	SeqMat* mat = (SeqMat*)A;
	if(mat)
		mat->op->matmultadd(alpha, A, B, beta, C, reuse);
}

void SeqMatMatTransposeMultAdd(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	SeqMat* mat = (SeqMat*)A;
	if(mat)
		mat->op->mattransposemultadd(alpha, A, B, beta, C, reuse);
}

void SeqMatGetDiag(void* A, value_type* diag, index_type bs) {
	SeqMat* mat = (SeqMat*)A;
	if(mat)
		mat->op->get_diag(A, diag, bs);
}

void SeqMatAddElemValueBatched(void* A,
																index_type nelem, index_type nshl,
																index_type *ien,
																byte* row_mask, byte* col_mask,
																value_type* value, Arena scratch) {
	SeqMat* mat = (SeqMat*)A;
	if(mat)
		mat->op->add_elem_value_batched(A, nelem, nshl, ien, row_mask, col_mask, value, scratch);
}

