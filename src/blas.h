#ifndef __BLAS_H__
#define __BLAS_H__

#include "common.h"

#ifdef CDAM_USE_CUDA
#include <cublas_v2.h>
typedef cublasOperation_t BLASTrans;
#define BLAS_T CUBLAS_OP_T
#define BLAS_N CUBLAS_OP_N
#define BLAS_C CUBLAS_OP_C

typedef cublasFillMode_t BLASUpLo;
#define BLAS_UP CUBLAS_FILL_MODE_UPPER
#define BLAS_LO CUBLAS_FILL_MODE_LOWER

typedef cublasDiagType_t BLASDiag;
#define BLAS_U CUBLAS_DIAG_UNIT
#define BLAS_NU CUBLAS_DIAG_NON_UNIT

#include <cusparse.h>
typedef cusparseOperation_t SPTrans;
#define SP_N CUSPARSE_OPERATION_NON_TRANSPOSE
#define SP_T CUSPARSE_OPERATION_TRANSPOSE
#define SP_C CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
typedef cusparseMatDescr_t SPMatDesc;


#elif defined(CDAM_USE_MKL)
#include "mkl.h"
#define BLASTrans CBLAS_TRANSPOSE
#define BLAS_T CblasTrans
#define BLAS_N CblasNoTrans
#define BLAS_C CblasConjTrans

#define BLASUpLo CBLAS_UPLO
#define BLAS_UP CblasUpper
#define BLAS_LO CblasLower

#define BLASDiag CBLAS_DIAG
#define BLAS_U CblasUnit
#define BLAS_NU CblasNonUnit

#include "mkl_spblas.h"
typedef sparse_operation_t SPTrans;
#define SP_N SPARSE_OPERATION_NON_TRANSPOSE
#define SP_T SPARSE_OPERATION_TRANSPOSE
#define SP_C SPARSE_OPERATION_CONJUGATE_TRANSPOSE
typedef sparse_matrix_t SPMatDesc;

#elif defined(CDAM_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
typedef enum {
	BLAS_T = CblasTrans,
	BLAS_N = CblasNoTrans,
	BLAS_C = CblasConjTrans
} BLASTrans;

typedef enum {
	BLAS_UP = CblasUpper,
	BLAS_LO = CblasLower
} BLASUpLo;

typedef enum {
	BLAS_U = CblasUnit,
	BLAS_NU = CblasNonUnit
} BLASDiag;

typedef enum {
	SP_N = CblasNoTrans,
	SP_T = CblasTrans
} SPTrans;

typedef sparse_matrix_double SPMatDesc;

#else 
#error "No BLAS library specified"
#endif

__BEGIN_DECLS__
/* BLAS Helper */
void SetPointerModeDevice();
void SetPointerModeHost();

/* BLAS Level 1 */
void dscal(int n, double alpha, double *x, int incx);
void dcopy(int n, double *x, int incx, double *y, int incy);
void ddot(int n, double *x, int incx, double *y, int incy, double *r);
void daxpy(int n, double alpha, double *x, int incx, double *y, int incy);
void dnrm2(int n, double *x, int incx, double *r);
void drot(int n, double *x, int incx, double *y, int incy, double c, double s);
void drotg(double a, double b, double *c, double *s);

/* BLAS Level 2 */
void dgemv(BLASTrans trans, int m, int n, double alpha, const double *A, int lda, const double *x, int incx, double beta, double *y, int incy);

void dtrsv(BLASUpLo uplo, BLASTrans trans, BLASDiag diag, int n, const double *A, int lda, double *x, int incx); 

/* BLAS Level 3 */
void dgemm(BLASTrans transA, BLASTrans transB, int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);

/* BLAS Extension */

void dgemvBatched(BLASTrans trans, int m, int n, double alpha,
									const double *const Aarray[], int lda,
									const double *const xarray[], int incx, double beta,
									double *const yarray[], int incy, int batchCount);

void dgemvStridedBatched(BLASTrans trans, int m, int n, double alpha,
												const double *A, int lda, int strideA,
												const double *x, int incx, int strideX, double beta,
												double *y, int incy, int strideY, int batchCount);

void dgemmBatched(BLASTrans transA, BLASTrans transB, int m, int n, int k, double alpha,
									const double *const Aarray[], int lda,
									const double *const Barray[], int ldb, double beta,
									double *const Carray[], int ldc, int batchCount);
void dgemmStridedBatched(BLASTrans transA, BLASTrans transB, int m, int n, int k, double alpha,
												const double *A, int lda, int strideA,
												const double *B, int ldb, int strideB, double beta,
												double *C, int ldc, int strideC, int batchCount);

/* LAPACK */
void dgetrfBatched(int n, double *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);
void dgetriBatched(int n, double *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);

/* Sparse BLAS */

void SpMatCreate(SPMatDesc* matDesc, int m, int n, int nnz, int* row_ptr, int* col_ind, double* values);
void SpMatDestroy(SPMatDesc matDesc);
void dspmvBufferSize(SPTrans trans, double alpha, SPMatDesc matA, const double* x, double beta, double* y, size_t* bufferSize);
void dspmvPreprocess(SPTrans trans, double alpha, SPMatDesc matA, const double* x, double beta, double* y, void* buffer);
void dspmv(SPTrans trans, double alpha, SPMatDesc matA, const double* x, double beta, double* y, void* buffer);

__END_DECLS__

#endif /* __BLAS_H__ */
