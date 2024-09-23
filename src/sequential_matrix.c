#include "alloc.h"
#include "blas.h"
#include "sequential_matrix.h"

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
	SeqMatZeroRowDenseGPU(order, data, nrow, ncol, row, rows, shift, 0, diag);
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

	if(typeB == Mat_DENSE) {
		SeqMatMatMultAddDenseDensePrivate(alpha, A, B, beta, C, reuse);
	}
	else if(typeB == Mat_CSR) {
		SeqMatMatMultAddDenseCSRPrivate(alpha, A, B, beta, C, reuse);
	}
	else {
		ASSERT(0 && "Unsupported matrix type.");
	}
}

static void SeqMatGetDiagDensePrivate(void* A, value_type* diag, index_type bs) {
	/* TODO: Implement this function */
}
static void SeqMatAddValueBatchedDensePrivate(void* A,
																							index_type batch_size, index_type* batch_index_ptr,
																							index_type *ien, index_type nshl,
																							index_type block_row, index_type block_col,
																							value_type* value, index_type ldv, index_type stride) {
	/* TODO: Implement this function */
}



