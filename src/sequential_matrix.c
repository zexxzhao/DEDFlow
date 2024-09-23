#include "alloc.h"
#include "blas.h"
#include "sequential_matrix.h"
#include "sequential_matrix_device.h"

static void SeqMatSetupDensePrivate(void* A) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	SeqMatAsType(A, SeqMatDense)->order = MAT_COL_MAJOR;

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
	MatOrder order = SeqMatAsType(A, SeqMatDense)->order;
	ASSERT(order == MAT_ROW_MAJOR && "Only row-major matrix is supported.");
#ifdef CDAM_USE_CUDA
	SeqMatZeroRowDenseGPU(order, data, nrow, ncol, row, rows, shift, 0, diag, 0);
#else
	index_type i, j, ir;
	if(order == MAT_ROW_MAJOR) {
		for(i = 0; i < row; i++) {
			ir = rows[i] + shift;
			if(ir >= 0 && ir < nrow) {
				CdamMemset(data + ir * ncol, 0, ncol * sizeof(value_type), DEVICE_MEM);
				if(ir < ncol) {
					CdamMemcpy(data + ir * ncol + ir, &diag, sizeof(value_type), DEVICE_MEM, HOST_MEM);
				}
			}
		}
	}
	else { /* MAT_COL_MAJOR */
		for(i = 0; i < row; i++) {
			ir = rows[i] + shift;
			if(ir >= 0 && ir < nrow) {
				for(j = 0; j < ncol; j++) {
					data[j * nrow + ir] = 0;
				}
				if(ir < ncol) {
					data[ir * nrow + ir] = diag;
				}
			}
		}
	}
#endif /* CDAM_USE_CUDA */
}

static void SeqMatGetSubmatDensePrivate(void* A, index_type nr, index_type *row,
																				index_type nc, index_type *col, void* B, void** auxiliary_data) {
	index_type i, j;
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);

	ASSERT(SeqMatType(A) == MAT_TYPE_DENSE && "Matrix A must be dense.");
	ASSERT(SeqMatType(B) == MAT_TYPE_DENSE && "Matrix B must be dense.");
	if(nr != SeqMatNumRow(B) || nc != SeqMatNumCol(B)) {
		SeqMatDestroyDensePrivate(B);
		SeqMatNumRow(B) = nr;
		SeqMatNumCol(B) = nc;
		SeqMatAsType(B, SeqMatDense)->order = SeqMatAsType(A, SeqMatDense)->order;
		SeqMatAsType(B, SeqMatDense) = CdamTMalloc(SeqMatDense, 1, HOST_MEM);
		CdamMemset(SeqMatAsType(B, SeqMatDense), 0, sizeof(SeqMatDense), HOST_MEM);
		SeqMatSetupDensePrivate(B);
	}

	value_type* dataA = SeqMatAsType(A, SeqMatDense)->data;
	value_type* dataB = SeqMatAsType(B, SeqMatDense)->data;
	*auxiliary_data = NULL;
#ifdef CDAM_USE_CUDA
	SeqMatGetSubmatDenseGPU(SeqMatAsType(B, SeqMatDense)->order, dataA, nrow, ncol, dataB, nr, nc, row, col, 0);
#else
	if(SeqMatAsType(A, SeqMatDense)->order == MAT_ROW_MAJOR) {
		for(i = 0; i < nr; i++) {
			for(j = 0; j < nc; j++) {
				dataB[i * nc + j] = dataA[row[i] * ncol + col[j]];
			}
		}
	}
	else {
		for(i = 0; i < nr; i++) {
			for(j = 0; j < nc; j++) {
				dataB[i + j * nr] = dataA[row[i] + col[j] * nrow];
			}
		}
	}
#endif /* CDAM_USE_CUDA */
}

static void SeqMatCopyDensePrivate(void* A, void* B, MatResue reuse) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);

	if(reuse == MAT_REUSE) {
		ASSERT(nrow == SeqMatNumRow(B) && ncol == SeqMatNumCol(B) && "Matrix size mismatch.");
		ASSERT(SeqMatAsType(A, SeqMatDense)->order == SeqMatAsType(B, SeqMatDense)->order && "Matrix order mismatch.");
		CdamMemcpy(SeqMatAsType(B, SeqMatDense)->data, SeqMatAsType(A, SeqMatDense)->data, nrow * ncol * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
	} else {
		SeqMatDestroyDensePrivate(B);
		SeqMatNumRow(B) = nrow;
		SeqMatNumCol(B) = ncol;
		SeqMatAsType(B, SeqMatDense)->order = SeqMatAsType(A, SeqMatDense)->order;
		SeqMatAsType(B, SeqMatDense) = CdamTMalloc(SeqMatDense, 1, HOST_MEM);
		CdamMemset(SeqMatAsType(B, SeqMatDense), 0, sizeof(SeqMatDense), HOST_MEM);
		SeqMatSetupDensePrivate(B);
		SeqMatCopyDensePrivate(A, B, MAT_REUSE);
	}
}

static void SeqMatTranspose(void *A, void* B) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	MatOrder order = SeqMatAsType(A, SeqMatDense)->order;

	ASSERT(A && "Matrix A is NULL.");

	if(B && A != B) {
		SeqMatNumRow(B) = ncol;
		SeqMatNumCol(B) = nrow;

		if(order == MAT_ROW_MAJOR) {
			SeqMatAsType(B, SeqMatDense)->order = MAT_COL_MAJOR;
		} else {
			SeqMatAsType(B, SeqMatDense)->order = MAT_ROW_MAJOR;
		}

		value_type* dataA = SeqMatAsType(A, SeqMatDense)->data;
		value_type* dataB = SeqMatAsType(B, SeqMatDense)->data;

		CdamMemcpy(dataB, dataA, nrow * ncol * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
	}
	else {
		if(order == MAT_ROW_MAJOR) {
			SeqMatAsType(A, SeqMatDense)->order = MAT_COL_MAJOR;
		} else {
			SeqMatAsType(A, SeqMatDense)->order = MAT_ROW_MAJOR;
		}
		SeqMatNumRow(A) = ncol;
		SeqMatNumCol(A) = nrow;
	}
}

static void SeqMatMultAddDensePrivate(value_type alpha, void* A, value_type *x, value_type beta, value_type *y) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	value_type* data = SeqMatAsType(A, SeqMatDense)->data;

	if(SeqMatAsType(A, SeqMatDense)->order == MAT_ROW_MAJOR) {
		dgemv(BLAS_T, ncol, nrow, alpha, data, ncol, x, 1, beta, y, 1);
	}
	else if(SeqMatAsType(A, SeqMatDense)->order == MAT_COL_MAJOR) {
		dgemv(BLAS_N, nrow, ncol, alpha, data, nrow, x, 1, beta, y, 1);
	}
}

static void SeqMatMatMultAddDenseDensePrivate(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	index_type m, n, k;
	value_type* dataA = SeqMatAsType(A, SeqMatDense)->data;
	value_type* dataB = SeqMatAsType(B, SeqMatDense)->data;
	value_type* dataC = SeqMatAsType(C, SeqMatDense)->data;

	BLASTrans transA = BLAS_T;
	if(SeqMatAsType(A, SeqMatDense)->order == MAT_COL_MAJOR) {
		transA = BLAS_N;
	}
	BLASTrans transB = BLAS_T;
	if(SeqMatAsType(B, SeqMatDense)->order == MAT_COL_MAJOR) {
		transB = BLAS_N;
	}

	m = (transA == BLAS_N) ? SeqMatNumRow(A) : SeqMatNumCol(A);
	n = (transB == BLAS_N) ? SeqMatNumCol(B) : SeqMatNumRow(B);
	k = (transA == BLAS_N) ? SeqMatNumCol(A) : SeqMatNumRow(A);


	ASSERT(m == SeqMatNumRow(C) && n == SeqMatNumCol(C) && "Matrix size mismatch.");
	ASSERT(k == (transB == BLAS_N) ? SeqMatNumRow(B) : SeqMatNumCol(B) && "Matrix size mismatch.");

	ASSERT(reuse == MAT_REUSE && "Only reuse is supported.");
	dgemm(transA, transB, m, n, k, alpha, dataA, SeqMatNumRow(A), dataB, SeqMatNumRow(B), beta, dataC, SeqMatNumRow(C));
}

static void SeqMatMatMultAddDenseCSRPrivate(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	/* AB= (AB)^T^T = (B^T A^T)^T */
	SeqMatTranspose(A, A);
	SeqMatTranspose(B, B);
	SeqMatMatMultAddCSRDense(alpha, B, A, beta, C, reuse);
	SeqMatTranspose(A, A);
	SeqMatTranspose(B, B);
	SeqMatTranspose(C, C);
};
static void SeqMatMatMultAddDensePrivate(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	MatType typeB = SeqMatType(B);

	if(typeB == MAT_TYPE_DENSE) {
		SeqMatMatMultAddDenseDensePrivate(alpha, A, B, beta, C, reuse);
	}
	else if(typeB == MAT_TYPE_CSR) {
		SeqMatMatMultAddDenseCSRPrivate(alpha, A, B, beta, C, reuse);
	}
	else {
		ASSERT(0 && "Unsupported matrix type.");
	}
}

static void SeqMatGetDiagDensePrivate(void* A, value_type* diag, index_type bs) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);

	ASSERT(nrow == ncol && "Matrix must be square.");
	ASSERT(nrow % bs == 0 && "Block size must be a divisor of the matrix size.");
#ifdef CDAM_USE_CUDA
	SeqMatGetDiagDenseGPU(SeqMatAsType(A, SeqMatDense)->data, n, diag, bs, 0);
#else
	index_type i;
	value_type* data = SeqMatAsType(A, SeqMatDense)->data;
	for(i = 0; i < nrow * bs; ++i) {
		diag[i] = data[(i / bs) * nrow + i / (bs * bs) * bs + i % bs];
	}
#endif
}

#define DOFMap(storage_order, nrow, ncol, i, j) ((storage_order == MAT_ROW_MAJOR) ? (i * ncol + j) : (i + j * nrow))

static void SeqMatAddElemValueBatchedDensePrivate(void* A,
																									index_type batch_size, index_type* batch_index_ptr,
																									index_type *ien, index_type nshl,
																									index_type block_row, index_type block_col,
																									value_type* value, index_type ldv, index_type stride) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	value_type* data = SeqMatAsType(A, SeqMatDense)->data;
	MatStorageMethod rmap_storage = SeqMatRowStorageMethod(A);
	MatStorageMethod cmap_storage = SeqMatColStorageMethod(A);
	/* Logically, shape(value) = (batch_size, nshl, nshl, block_row, block_col */
	/* Physically, shape(value) = (batch_size, nshl, nshl, stride) */
	/* stride >= block_row * ldv && ldv > block_col */
	/* Each block in value is stored in a row-major order. */

#ifdef CDAM_USE_CUDA
	SeqMatAddElemValueBatchedDenseGPU(data, nrow, ncol,
																		SeqMatAsType(A, SeqMatDense)->order, rmap_storage, cmap_storage,
																		batch_size, batch_index_ptr, ien, nshl,
																		block_row, block_col, value, ldv, stride, 0);
#else
	index_type i, ir, ic, iel;
	index_type batch, ishl, jshl, ishg, jshg;
	index_type dst_row, dst_col;
	value_type* value_ptr;
	for(i = 0; i < batch_size * nshl * nshl; ++i) {
		batch = i / (nshl * nshl);
		iel = batch_index_ptr[batch];
		ishl = (i % (nshl * nshl)) / nshl;
		jshl = i % nshl;
		ishg = ien[iel * nshl + ishl];
		jshg = ien[iel * nshl + jshl];

		value_ptr = value + i * stride;

		for(ir = 0; ir < block_row; ir++) {
			dst_row = DOFMap(rmap_storage, nrow / block_row, block_row, ishg, ir);
			for(ic = 0; ic < block_col; ++ic) {
				dst_col = DOFMap(cmap_storage, ncol / block_col, block_col, jshg, ic);
				data[DOFMap(SeqMatAsType(A, SeqMatDense)->order, nrow, ncol, dst_row, dst_col)] += value_ptr[ir * ldv + ic];
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

	SpMatCreate(SeqMatAsType(A, SeqMatCSR)->descr, nrow, ncol, spy->nnz, CSRAttrRowPtr(spy), CSRAttrColInd(spy), SeqMatAsType(A, SeqMatCSR)->data);

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
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	value_type* data = SeqMatAsType(A, SeqMatCSR)->data;
	CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;
	CdamMemset(data, 0, spy->nnz * sizeof(value_type), DEVICE_MEM);
}

static void SeqMatZeroRowCSRPrivate(void* A, index_type row, index_type* rows, index_type shift, value_type diag) {
	CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	index_type* row_ptr = CSRAttrRowPtr(spy);
	index_type* col_ind = CSRAttrColInd(spy);
	value_type* data = SeqMatAsType(A, SeqMatCSR)->data;

#ifdef CDAM_USE_CUDA
	SeqMatZeroRowCSRGPU(data, nrow, ncol, row_ptr, col_ind, row, rows, shift, diag, 0);
#else
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
																			index_type nc, index_type* col, void* B, void** auxiliary_data) {
	index_type i, j;
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;
	value_type* dataA = SeqMatAsType(A, SeqMatCSR)->data;
	CSRAttr* new_spy = NULL;

	ASSERT(SeqMatType(A) == MAT_TYPE_CSR && "Matrix A must be CSR.");
	ASSERT(SeqMatType(B) == MAT_TYPE_CSR && "Matrix B must be CSR.");

	if(nr != SeqMatNumRow(B) || nc != SeqMatNumCol(B)) {
		SeqMatDestroyCSRPrivate(B);
		SeqMatNumRow(B) = nr;
		SeqMatNumCol(B) = nc;
		SeqMatRowStorageMethod(B) = SeqMatRowStorageMethod(A);
		SeqMatColStorageMethod(B) = SeqMatColStorageMethod(A);
		GenerateSubmatCSRAttr(spy, nr, row, nc, col, &new_spy);
		SeqMatAsType(B, SeqMatCSR)->spy = new_spy;
		SeqMatSetupCSRPrivate(B);
	}
#ifdef CDAM_USE_CUDA
	SeqMatCopySubmatValueCSRGPU(dataA, SeqMatAsType(A, SeqMatCSR)->spy, nr, row, nc, col, SeqMatAsType(B, SeqMatCSR)->data, SeqMatAsType(A, SeqMatCSR)->spy, 0);
#else
	value_type* dataB = SeqMatAsType(B, SeqMatCSR)->data;
	index_type ic;
	for(i = 0; i < nr; i++) {
		ir = row[i];
		for(j = spy->row_ptr[ir]; j < spy->row_ptr[ir + 1]; j++) {
			ic = spy->col_ind[j];
			for(k = 0; k < nc; ++k) {
				if(col[k] == ic) {
					dataB[new_spy->row_ptr[i] + k] = dataA[j];
					break;
				}
			}
		}
	}
#endif /* CDAM_USE_CUDA */
	*auxiliary_data = new_spy;
}

static void SeqMatCopyCSRPrivate(void* A, void* B) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;

	if(SeqMatNumRow(B) != nrow || SeqMatNumCol(B) != ncol) {
		SeqMatDestroyCSRPrivate(B);
		SeqMatNumRow(B) = nrow;
		SeqMatNumCol(B) = ncol;
		SeqMatRowStorageMethod(B) = SeqMatRowStorageMethod(A);
		SeqMatColStorageMethod(B) = SeqMatColStorageMethod(A);
		SeqMatAsType(B, SeqMatCSR)->spy = spy;
		SeqMatSetupCSRPrivate(B);
	}
	CdamMemcpy(SeqMatAsType(B, SeqMatCSR)->data, SeqMatAsType(A, SeqMatCSR)->data, spy->nnz * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
}

static void SeqMatTransposeCSRPrivate(void* A, void* B) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;

	ASSERT(A && "Matrix A is NULL.");

	if(B && A != B) {
		/* TODO: Implement this */
		// SeqMatNumRow(B) = ncol;
		// SeqMatNumCol(B) = nrow;
		// SeqMatRowStorageMethod(B) = SeqMatColStorageMethod(A);
		// SeqMatColStorageMethod(B) = SeqMatRowStorageMethod(A);
		// GenerateTransposeCSRAttr(spy, &SeqMatAsType(B, SeqMatCSR)->spy);
		// SeqMatSetupCSRPrivate(B);
		// SeqMatTransposeCSRPrivate(A, B);
	}
	else {
		/* TODO: Implement this */
		// GenerateTransposeCSRAttr(spy, &SeqMatAsType(A, SeqMatCSR)->spy);
		// SeqMatSetupCSRPrivate(A);
	}
}

static void SeqMatMultAddCSRPrivate(value_type alpha, void* A, value_type* x, value_type beta, value_type* y) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);
	CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;
	value_type* data = SeqMatAsType(A, SeqMatCSR)->data;

	dspmv(SP_N, alpha, SeqMatAsType(A, SeqMatCSR)->descr, x, beta, y, data, SeqMatAsType(A, SeqMatCSR)->buffer);
}

static void SeqMatMatMultAddCSRPrivate(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	/* TODO: Wrap those functions in blas.h */
	/* CSR * CSR => SpGEMM or SpGEMMreuse */
	/* CSR * Dense => SpMM */
}

static void SeqMatGetDiagCSRPrivate(void* A, value_type* diag, index_type bs) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);

	ASSERT(nrow == ncol && "Matrix must be square.");

	if(bs >= 1) {
#ifdef CDAM_USE_CUDA
		SeqMatGetDiagBlockCSRGPU(SeqMatAsType(A, SeqMatCSR)->data, SeqMatAsType(A, SeqMatCSR)->spy, diag, bs, bs * bs, bs, 0);
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
																								index_type batch_size, index_type* batch_index_ptr,
																								index_type *ien, index_type nshl,
																								index_type block_row, index_type block_col,
																								value_type* value, index_type ldv, index_type stride) {
	index_type nrow = SeqMatNumRow(A);
	index_type ncol = SeqMatNumCol(A);

	CSRAttr* spy = SeqMatAsType(A, SeqMatCSR)->spy;
	value_type* data = SeqMatAsType(A, SeqMatCSR)->data;
	MatStorageMethod rmap_storage = SeqMatRowStorageMethod(A);
	MatStorageMethod cmap_storage = SeqMatColStorageMethod(A);

#ifdef CDAM_USE_CUDA
	SeqMatAddElemValueBatchedCSRGPU(1.0, data, spy, rmap_storage, cmap_storage,
																	batch_size, batch_index_ptr, ien, nshl,
																	block_row, block_col, value, ldv, stride, 0);
#else
	index_type i, ir, ic, iel;
	index_type batch, ishl, jshl, ishg, jshg;
	index_type dst_row, dst_col;
	index_type start, end;
	value_type* value_ptr;
	for(i = 0; i < batch_size * nshl * nshl; ++i) {
		batch = i / (nshl * nshl);
		iel = batch_index_ptr[batch];
		ishl = (i % (nshl * nshl)) / nshl;
		jshl = i % nshl;
		ishg = ien[iel * nshl + ishl];
		jshg = ien[iel * nshl + jshl];

		value_ptr = value + i * stride;

		for(ir = 0; ir < block_row; ir++) {
			dst_row = DOFMap(rmap_storage, nrow / block_row, block_row, ishg, ir);
			start = CSRAttrRowPtr(spy)[dst_row];
			end = CSRAttrRowPtr(spy)[dst_row + 1];
			for(ic = 0; ic < block_col; ++ic) {
				dst_col = DOFMap(cmap_storage, ncol / block_col, block_col, jshg, ic);
				for(j = start; j < end; j++) {
					if(CSRAttrColInd(spy)[j] == dst_col) {
						data[j] += value_ptr[ir * ldv + ic];
						break;
					}
				}
			}
		}
	}
#endif
}

void SeqMatCreate(MatType type, index_type nrow, index_type ncol, void** A) {
	*A = CdamTMalloc(SeqMat, 1, HOST_MEM);
	SeqMat* mat = *A;
	CdamMemset(*A, 0, sizeof(SeqMat), HOST_MEM);
	SeqMatType(*A) = type;
	SeqMatNumRow(*A) = nrow;
	SeqMatNumCol(*A) = ncol;
	SeqMatRowStorageMethod(*A) = MAT_COL_MAJOR; /* Default */
	SeqMatColStorageMethod(*A) = MAT_COL_MAJOR;
	if(type == MAT_TYPE_DENSE) {
		SeqMatAsType(*A, SeqMatDense) = CdamTMalloc(SeqMatDense, 1, HOST_MEM);
		CdamMemset(SeqMatAsType(*A, SeqMatDense), 0, sizeof(SeqMatDense), HOST_MEM);
		mat->op->setup = SeqMatSetupDensePrivate;
		mat->op->destroy = SeqMatDestroyDensePrivate;
		mat->op->zero = SeqMatZeroDensePrivate;
		mat->op->zero_row = SeqMatZereRowDensePrivate;
		mat->op->get_submat = SeqMatGetSubmatDensePrivate;
		mat->op->copy = SeqMatCopyDensePrivate;
		mat->op->transpose = SeqMatTranspose;
		mat->op->mult_add = SeqMatMultAddDensePrivate;
		mat->op->mat_mult_add = SeqMatMatMultAddDensePrivate;
		mat->op->get_diag = SeqMatGetDiagDensePrivate;
		mat->op->add_elem_value_batched = SeqMatAddElemValueBatchedDensePrivate;
	}
	else if(type == MAT_TYPE_CSR) {
		SeqMatAsType(*A, SeqMatCSR) = CdamTMalloc(SeqMatCSR, 1, HOST_MEM);
		CdamMemset(SeqMatAsType(*A, SeqMatCSR), 0, sizeof(SeqMatCSR), HOST_MEM);
		mat->op->setup = SeqMatSetupCSRPrivate;
		mat->op->destroy = SeqMatDestroyCSRPrivate;
		mat->op->zero = SeqMatZeroCSRPrivate;
		mat->op->zero_row = SeqMatZeroRowCSRPrivate;
		mat->op->get_submat = SeqMatGetSubmatCSRPrivate;
		mat->op->copy = SeqMatCopyCSRPrivate;
		mat->op->transpose = SeqMatTransposeCSRPrivate;
		mat->op->mult_add = SeqMatMultAddCSRPrivate;
		mat->op->mat_mult_add = SeqMatMatMultAddCSRPrivate;
		mat->op->get_diag = SeqMatGetDiagCSRPrivate;
		mat->op->add_elem_value_batched = SeqMatAddElemValueBatchedCSRPrivate;
	}
	else {
		ASSERT(0 && "Unsupported matrix type.");
	}
}

void SeqMatSetup(void* A) {
	mat->op->setup(A);
}
void SeqMatDestroy(void* A) {
	mat->op->destroy(A);
	CdamFree(A, sizeof(SeqMat), HOST_MEM);
}

void SeqMatZero(void* A) {
	mat->op->zero(A);
}

void SeqMatZeroRow(void* A, index_type row, index_type* rows, index_shift shift, value_type diag) {
	mat->op->zero_row(A, row, rows, shift, diag);
}

void SeqMatGetSubmat(void* A, index_type nr, index_type* rows, index_type nc, index_type* cols, void* B, void** auxiliary_data) {
	mat->op->get_submat(A, nr, rows, nc, cols, B, auxiliary_data);
}

void SeqMatCopy(void* A, void* B) {
	mat->op->copy(A, B);
}
void SeqMatTranspose(void* A, void* B) {
	mat->op->transpose(A, B);
}

void SeqMatMultAdd(value_type alpha, void* A, value_type* x, value_type beta, value_type* y) {
	mat->op->mult_add(alpha, A, x, beta, y);
}
void SeqMatMatMultAdd(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	mat->op->mat_mult_add(alpha, A, B, beta, C, reuse);
}

void SeqMatGetDiag(void* A, value_type* diag, index_type bs) {
	mat->op->get_diag(A, diag, bs);
}
void SeqMatAddElemValueBatched(void* A, index_type batch_size, index_type* ien, index_type nshl,
															 index_type block_row, index_type block_col,
															 value_type* value, index_type ldv, index_type stride) {
	mat->op->add_elem_value_batched(A, batch_size, ien, nshl, block_row, block_col, value, ldv, stride);
}


