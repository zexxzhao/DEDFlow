#ifndef __BLAS_H__
#define __BLAS_H__

#include "common.h"

#ifdef CDAM_USE_CUDA
#include <cublas_v2.h>
#define BLASTrans cublasOperation_t
#define BLAS_T CUBLAS_OP_T
#define BLAS_N CUBLAS_OP_N
#define BLAS_C CUBLAS_OP_C

#define BLASUpLo cublasFillMode_t
#define BLAS_UP CUBLAS_FILL_MODE_UPPER
#define BLAS_LO CUBLAS_FILL_MODE_LOWER

#define BLASDiag cublasDiagType_t
#define BLAS_U CUBLAS_DIAG_UNIT
#define BLAS_NU CUBLAS_DIAG_NON_UNIT

#else
#include <cblas.h>
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

#endif

__BEGIN_DECLS__

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

__END_DECLS__

#endif /* __BLAS_H__ */
