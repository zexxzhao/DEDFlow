#include "common.h"
#include "blas.h"

#ifdef CDAM_USE_CUDA

static cublasHandle_t GetCublasHandle() {
	return *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE);
}

static cusparseHandle_t GetCusparseHandle() {
	return *(cusparseHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUSPARSE_HANDLE);
}

/* BLAS Level 1 */
void dscal(int n, double alpha, double *x, int incx) {
	cublasDscal(GetCublasHandle(), n, &alpha, x, incx);	
}
void dcopy(int n, double *x, int incx, double *y, int incy) {
	cublasDcopy(GetCublasHandle(), n, x, incx, y, incy);
}
void ddot(int n, double *x, int incx, double *y, int incy, double *r) {
	cublasDdot(GetCublasHandle(), n, x, incx, y, incy, r);
}
void daxpy(int n, double alpha, double *x, int incx, double *y, int incy) {
	cublasDaxpy(GetCublasHandle(), n, &alpha, x, incx, y, incy);
}
void dnrm2(int n, double *x, int incx, double *r) {
	cublasDnrm2(GetCublasHandle(), n, x, incx, r);
}
void drot(int n, double *x, int incx, double *y, int incy, double c, double s) {
	cublasDrot(GetCublasHandle(), n, x, incx, y, incy, &c, &s);
}
void drotg(double a, double b, double *c, double *s) {
	cublasDrotg(GetCublasHandle(), &a, &b, c, s);
}

/* BLAS Level 2 */
void dgemv(BLASTrans trans, int m, int n,
					 double alpha,
					 const double *A, int lda,
					 const double *x, int incx,
					 double beta,
					 double *y, int incy) {
	cublasDgemv(GetCublasHandle(), trans, m, n,
							&alpha, A, lda, x, incx, &beta, y, incy);
}

void dtrsv(BLASUpLo uplo, BLASTrans trans, BLASDiag diag,
					 int n, const double *A, int lda, double *x, int incx) {
	cublasDtrsv(GetCublasHandle(), uplo, trans, diag, n, A, lda, x, incx);
}

/* BLAS Level 3 */
void dgemm(BLASTrans transA, BLASTrans transB, int m, int n, int k,
					 double alpha, const double *A, int lda, const double *B, int ldb,
					 double beta, double *C, int ldc) {
	cublasDgemm(GetCublasHandle(), transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

/* BLAS Extension */

void dgemvBatched(BLASTrans trans, int m, int n, double alpha,
									const double *const Aarray[], int lda,
									const double *const xarray[], int incx, double beta,
									double *const yarray[], int incy, int batchCount) {
	cublasDgemvBatched(GetCublasHandle(), trans, m, n, &alpha, Aarray, lda, xarray, incx, &beta, yarray, incy, batchCount);
}

void dgemvStridedBatched(BLASTrans trans, int m, int n, double alpha,
												const double *A, int lda, int strideA,
												const double *x, int incx, int strideX, double beta,
												double *y, int incy, int strideY, int batchCount) {
	cublasDgemvStridedBatched(GetCublasHandle(), trans, m, n, &alpha, A, lda, strideA, x, incx, strideX, &beta, y, incy, strideY, batchCount);
}

void dgemmBatched(BLASTrans transA, BLASTrans transB, int m, int n, int k, double alpha,
									const double *const Aarray[], int lda,
									const double *const Barray[], int ldb, double beta,
									double *const Carray[], int ldc, int batchCount) {
	cublasDgemmBatched(GetCublasHandle(), transA, transB, m, n, k, &alpha, Aarray, lda, Barray, ldb, &beta, Carray, ldc, batchCount);
}
void dgemmStridedBatched(BLASTrans transA, BLASTrans transB, int m, int n, int k, double alpha,
												const double *A, int lda, int strideA,
												const double *B, int ldb, int strideB, double beta,
												double *C, int ldc, int strideC, int batchCount) {
	cublasDgemmStridedBatched(GetCublasHandle(), transA, transB, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);


/* LAPACK */
void dgetrfBatched(int n, double *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
	cublasDgetrfBatched(GetCublasHandle(), n, Aarray, lda, PivotArray, infoArray, batchSize);
}
void dgetriBatched(int n, double *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
	cublasDgetriBatched(GetCublasHandle(), n, Aarray, lda, PivotArray, infoArray, batchSize);
}

/* Sparse BLAS */

void SpMatCreate(SPMatDesc* matDesc, int m, int n, int* row_ptr, int* col_ind, double* values) {
	cusparseCreateCsr(matDesc, m, n, row_ptr, col_ind, values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
}	

void SpMatDestroy(SPMatDesc matDesc) {
	cusparseDestroyCsr(matDesc);
}

void dspmvBufferSize(SPTrans trans, double alpha, SPMatDesc matA, const double* x, double beta, double* y, size_t* bufferSize) {
	cusparseDnVecDescr_t xDescr, yDescr;
	i64 n_row, n_col;
	cusparseSpMatGetSize(matA, &n_row, &n_col);
	cusparseCreateDnVec(&xDescr, n_col, x);
	cusparseCreateDnVec(&yDescr, n_row, y);

	cusparseSPMV_bufferSize(GetCusparseHandle(), trans,
													&alpha, matA, xDescr, &beta, yDescr,
													CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, bufferSize);

	cusparseDestroyDnVec(xDescr);
	cusparseDestroyDnVec(yDescr);
}

void dspmvPreprocess(SPTrans trans, double alpha, SPMatDesc matA, const double* x, double beta, double* y, void* buffer) {
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION > 12300
	cusparseDnVecDescr_t xDescr, yDescr;
	i64 n_row, n_col;
	cusparseSpMatGetSize(matA, &n_row, &n_col);
	cusparseCreateDnVec(&xDescr, n_col, x);
	cusparseCreateDnVec(&yDescr, n_row, y);

	cusparseSPMV_preprocess(GetCusparseHandle(), trans,
													&alpha, matA, xDescr, &beta, yDescr,
													CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);

	cusparseDestroyDnVec(xDescr);
	cusparseDestroyDnVec(yDescr);
#endif
}

void dspmv(SPTrans trans, double alpha, SPMatDesc matA, const double* x, double beta, double* y, void* buffer) {
	cusparseDnVecDescr_t xDescr, yDescr;
	i64 n_row, n_col;
	cusparseSpMatGetSize(matA, &n_row, &n_col);
	cusparseCreateDnVec(&xDescr, n_col, x);
	cusparseCreateDnVec(&yDescr, n_row, y);

	cusparseSPMV(GetCusparseHandle(), trans,
							 &alpha, matA, xDescr, &beta, yDescr,
							 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);

	cusparseDestroyDnVec(xDescr);
	cusparseDestroyDnVec(yDescr);
}


#elif defined(CDAM_USE_MKL)

/* BLAS Level 1 */
void dscal(int n, double alpha, double *x, int incx) {
	cblas_dscal(n, alpha, x, incx);
}

void dcopy(int n, double *x, int incx, double *y, int incy) {
	cblas_dcopy(n, x, incx, y, incy);
}

void ddot(int n, double *x, int incx, double *y, int incy, double *r) {
	*r = cblas_ddot(n, x, incx, y, incy);
}

void daxpy(int n, double alpha, double *x, int incx, double *y, int incy) {
	cblas_daxpy(n, alpha, x, incx, y, incy);
}

void dnrm2(int n, double *x, int incx, double *r) {
	*r = cblas_dnrm2(n, x, incx);
}

void drot(int n, double *x, int incx, double *y, int incy, double c, double s) {
	cblas_drot(n, x, incx, y, incy, c, s);
}

void drotg(double a, double b, double *c, double *s) {
	cblas_drotg(&a, &b, c, s);
}

/* BLAS Level 2 */
void dgemv(BLASTrans trans, int m, int n,
					 double alpha,
					 const double *A, int lda,
					 const double *x, int incx,
					 double beta,
					 double *y, int incy) {
	cblas_dgemv(CblasColMajor, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void dtrsv(BLASUpLo uplo, BLASTrans trans, BLASDiag diag,
					 int n, const double *A, int lda, double *x, int incx) {
	cblas_dtrsv(CblasColMajor, uplo, trans, diag,
							n, A, lda, x, incx);
}

/* BLAS Level 3 */
void dgemm(BLASTrans transA, BLASTrans transB, int m, int n, int k,
					 double alpha, const double *A, int lda, const double *B, int ldb,
					 double beta, double *C, int ldc) {
	cblas_dgemm(CblasColMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

/* BLAS Extension */
void dgemvBatched(BLASTrans trans, int m, int n, double alpha,
									const double *const Aarray[], int lda,
									const double *const xarray[], int incx, double beta,
									double *const yarray[], int incy, int batchCount) {
	int i;
	for (i = 0; i < batchCount; i++) {
		dgemv(trans, m, n, alpha, Aarray[i], lda, xarray[i], incx, beta, yarray[i], incy);
	}
}

void dgemvStridedBatched(BLASTrans trans, int m, int n, double alpha,
												const double *A, int lda, int strideA,
												const double *x, int incx, int strideX, double beta,
												double *y, int incy, int strideY, int batchCount) {
	int i;
	for (i = 0; i < batchCount; i++) {
		dgemv(trans, m, n, alpha, A + i * strideA, lda, x + i * strideX, incx, beta, y + i * strideY, incy);
	}
}

void dgemmBatched(BLASTrans transA, BLASTrans transB, int m, int n, int k, double alpha,
									const double *const Aarray[], int lda,
									const double *const Barray[], int ldb, double beta,
									double *const Carray[], int ldc, int batchCount) {
	int i;
	for (i = 0; i < batchCount; i++) {
		dgemm(transA, transB, m, n, k, alpha, Aarray[i], lda, Barray[i], ldb, beta, Carray[i], ldc);
	}
}
void dgemmStridedBatched(BLASTrans transA, BLASTrans transB, int m, int n, int k, double alpha,
												const double *A, int lda, int strideA,
												const double *B, int ldb, int strideB, double beta,
												double *C, int ldc, int strideC, int batchCount) {
	int i;
	for (i = 0; i < batchCount; i++) {
		dgemm(transA, transB, m, n, k, alpha, A + i * strideA, lda, B + i * strideB, ldb, beta, C + i * strideC, ldc);
	}
}

/* LAPACK */
void dgetrfBatched(int n, double *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
	int i;
	for (i = 0; i < batchSize; i++) {
		infoArray[i] = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, Aarray[i], lda, PivotArray + i * n);
	}
}

void dgetriBatched(int n, double *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
	int i;
	for (i = 0; i < batchSize; i++) {
		infoArray[i] = LAPACKE_dgetri(LAPACK_COL_MAJOR, n, Aarray[i], lda, PivotArray + i * n);
	}
}

/* Sparse BLAS */
void SpMatCreate(SPMatDesc* matDesc, int m, int n, int* row_ptr, int* col_ind, double* values) {
	mkl_sparse_d_create_csr(matDesc, SPARSE_INDEX_BASE_ZERO, m, n, row_ptr, row_ptr + 1, col_ind, values);
}

void SpMatDestroy(SPMatDesc matDesc) {
	mkl_sparse_destroy(matDesc);
}

void dspmvBufferSize(SPTrans trans, double alpha, SPMatDesc matA, const double* x, double beta, double* y, size_t* bufferSize) {
	*bufferSize = 0;
}

void dspmv(SPTrans trans, double alpha, SPMatDesc matA, const double* x, double beta, double* y, void* buffer) {
	mkl_sparse_d_mv(trans, alpha, matA, x, beta, y);
}

#endif /* CDAM_USE_CUDA */
