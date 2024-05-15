#include <float.h>
#include <cublas_v2.h>
#include "alloc.h"

#include "csr.h"
#include "vec.h"
#include "matrix.h"
#include "krylov.h"

__BEGIN_DECLS__

static Krylov* KryloveInitPrivate(u32 max_iter, f64 atol, f64 rtol) {
	Krylov* ksp = (Krylov*)CdamMallocHost(sizeof(Krylov));
	ksp->max_iter = max_iter;
	ksp->atol = atol;
	ksp->rtol = rtol;

	ksp->ksp_solve = NULL;
	ksp->pc_apply = NULL;
	ksp->ksp_ctx = NULL;
	ksp->pc_ctx = NULL;
	return ksp;
}

static void PCJacobiApply(Matrix* A, f64* x, f64* y, void* ctx) {
	u32 n = MatrixNumRow(A);
	Krylov* ksp = (Krylov*)ctx;
	if(ksp->pc_ctx == NULL) {
		ksp->pc_ctx_size = n * sizeof(f64);
		ksp->pc_ctx = CdamMallocDevice(ksp->pc_ctx_size);
	}
	else if(ksp->pc_ctx_size != n * sizeof(f64)) {
		CdamFreeDevice(ksp->pc_ctx, ksp->pc_ctx_size);
		ksp->pc_ctx_size = n * sizeof(f64);
		ksp->pc_ctx = CdamMallocDevice(ksp->pc_ctx_size);
	}

	f64* diag = (f64*)ksp->pc_ctx;
	MatrixGetDiag(A, diag);
	VecPointwiseDiv(y, x, diag, n);
}
static void CGSolvePrivate(Matrix* A, f64* x, f64* b, void* ctx) {
	u32 n = MatrixNumRow(A);
	cusparseDnVecDescr_t vec_x, vec_b;
	cusparseCreateDnVec(&vec_x, n, x, CUDA_R_64F);
	cusparseCreateDnVec(&vec_b, n, b, CUDA_R_64F);

	// cusparseSpMatDescr_t mat_A = CSRMatrixDescr(A);
	// UNUSED(mat_A);
	/* TODO: Implement CG solver */
}

void GMRESResidualUpdatePrivate(f64*, f64*);

/* GMRES solver */
static void GMRESSolvePrivate(Matrix* A, f64* x, f64* b, void* ctx) {
	const f64 minus_one = -1.0, one = 1.0, zero = 0.0;
	Krylov* ksp = (Krylov*)ctx;
	u32 maxit = ksp->max_iter;
	f64 atol = ksp->atol;
	f64 rtol = ksp->rtol;
	void* buffer_mv = ksp->ksp_ctx;
	u32 iter;
	u32 n = MatrixNumRow(A);
	f64 rnrm_init, rnrm;
	b32 converged;

	// cusparseHandle_t cusparse_handle = ksp->handle;
	// cusparseDnVecDescr_t vec_r, vec_x, vec_tmp;
	// cusparseSpMatDescr_t mat_A = CSRMatrixDescr(A);

	cublasHandle_t cublas_handle;

#define QCOL(col) (Q + (col) * n)
#define HCOL(col) (H + (col) * (maxit + 1))
#define GVCOS(i) (gv + 2 * (i))
#define GVSIN(i) (gv + 2 * (i) + 1)

	/* Allocate memory for Q[n, maxit+1] */
	f64* Q = (f64*)CdamMallocDevice(n * sizeof(64) * (maxit + 1));
	cudaMemset(Q, 0, n * sizeof(f64) * maxit);
	/* Allocate memeory for Hessenberg matrix H[(maxit + 1) * maxit] */
	f64* H = (f64*)CdamMallocDevice((maxit + 1) * maxit * sizeof(f64));
	cudaMemset(H, 0, (maxit + 1) * maxit * sizeof(f64));
	/* Allocate memory for tmp[n] */
	f64* tmp = (f64*)CdamMallocDevice(n * sizeof(f64));
	cudaMemset(tmp, 0, n * sizeof(f64));
	/* Allocate memory for Givens rotation */
	/* gv[::2] = cs, gv[1::2] = sn */
	f64* gv = (f64*)CdamMallocDevice(2 * maxit * sizeof(f64));
	cudaMemset(gv, 0, 2 * maxit * sizeof(f64));
	/* Allocate memory for residual vector */
	f64* beta = (f64*)CdamMallocDevice((maxit + 1) * sizeof(f64));
	cudaMemset(beta, 0, (maxit + 1) * sizeof(f64));
	cublasSetVector(1, sizeof(f64), &one, 1, beta, 1);

	cublasCreate(&cublas_handle);
	// cusparseCreateDnVec(&vec_x, n, x, CUDA_R_64F);				/* vec_x -> x */
	// cusparseCreateDnVec(&vec_r, n, QCOL(0), CUDA_R_64F);  /* vec_r -> Q[:, 0] */
	// cusparseCreateDnVec(&vec_tmp, n, tmp, CUDA_R_64F);		/* vec_tmp -> tmp */
	/* 0. r = b - A * x */
	/* 0.0. r = b */
	cublasDcopy(cublas_handle, n, b, 1, QCOL(0), 1);
	/* 0.1. r -= A * x */
	MatrixAMVPBY(-1.0, A, x, 1.0, QCOL(0));
	// cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
	// 						 &minus_one, mat_A, vec_x, &one, vec_r,
	// 						 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer_mv);


	converged = FALSE;
	cublasDnrm2(cublas_handle, n, QCOL(0), 1, &rnrm_init);
	if (rnrm_init < atol) {
		converged = TRUE;
	}

	/* 1. Generate initial vector Q[:, 0] */
	/* Normalize Q[:, 0] */
	rnrm = (f64)1.0 / rnrm_init;
	cublasDscal(cublas_handle, n, &rnrm, QCOL(0), 1);


	iter = 0;
	while (!converged && iter < maxit) {
		/* 2. Q[:, iter + 1] = A * inv(P) * Q[:, iter] */
		/* 2.0. Apply preconditioner: tmp[:] = inv(P) Q[:, iter] */
		ksp->pc_apply(A, QCOL(iter), tmp, ksp);
		/* 2.1 let vec_r -> Q[:, iter + 1] */
		// cusparseDestroyDnVec(vec_r);
		// cusparseCreateDnVec(&vec_r, n, QCOL(iter + 1), CUDA_R_64F);
		/* 2.2. vec_r = A * tmp */
		MatrixMatVec(A, tmp, QCOL(iter + 1));
		// cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		// 						 &one, mat_A, vec_tmp, &zero, vec_r,
		// 						 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer_mv);

		/* 3. Arnoldi process */
		/* 3.1. H[0:iter+1, iter] = Q[0:n, 0:iter+1].T * Q[0:n, iter+1] */
		cublasDgemv(cublas_handle, CUBLAS_OP_T,
								n, iter + 1, &one, Q, n,
								QCOL(iter + 1), 1, &zero, 
								HCOL(iter), 1);
		/* 3.2. Q[0:n, iter+1] -= Q[0:n, 0:iter+1] * H[0:iter+1, iter] */
		cublasDgemv(cublas_handle, CUBLAS_OP_N,
							  n, iter + 1, &minus_one, Q, n,
								HCOL(iter), 1, &one,
								QCOL(iter + 1), 1);
		/* 3.3. H[iter+1, iter] = || Q[0:n, iter+1] || */
		cublasDnrm2(cublas_handle, n, QCOL(iter + 1), 1, HCOL(iter) + iter + 1);
		/* 3.4. Q[0:n, iter+1] /= H[iter+1, iter] */
		cublasGetVector(1, sizeof(f64), HCOL(iter) + iter, 1, &rnrm, 1);
		rnrm = (f64)1.0 / rnrm;
		cublasDscal(cublas_handle, n, &rnrm, QCOL(iter + 1), 1);

		/* 4. Apply Givens rotation to H[0:iter+2, iter] */
		/* 4.0. Apply Givens rotation to H[0:iter+1, iter] using cs[0:iter+1] and sn[0:iter+1] */
		for (u32 i = 0; i < iter; i++) {
			cublasDrot(cublas_handle, 1,
								 HCOL(iter) + i, 1,
								 HCOL(iter) + i + 1, 1,
								 GVCOS(i), GVSIN(i));
		}
		/* 4.1. Compute Givens rotation */
		cublasDrotg(cublas_handle, HCOL(iter) + iter, HCOL(iter) + iter + 1, GVCOS(iter), GVSIN(iter));
		/* 4.2. Apply Givens rotation to H[iter:iter+2, iter] */
		cublasDrot(cublas_handle, 1,
							 HCOL(iter) + iter, 1,
							 HCOL(iter) + iter + 1, 1,
							 GVCOS(iter), GVSIN(iter));
		/* 4.3. beta[iter+1] = -sn[iter] * beta[iter]
		 *			beta[iter+0] =  cn[iter] * beta[iter] */
		GMRESResidualUpdatePrivate(beta, GVCOS(iter));
		
		/* 4. Check convergence */
		cublasGetVector(1, sizeof(f64), beta + iter + 1, 1, &rnrm, 1);
		rnrm = fabs(rnrm);
		if (rnrm < atol || rnrm < rtol * (rnrm_init + DBL_EPSILON)) {
			converged = TRUE;
		}
		iter++;
	}
	/* 5. Solve linear system */
	/* 5.1. Solve upper triangular system H[0:iter+1, 0:iter] y = beta[0:iter+1] */
	cublasDtrsv(cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
							iter + 1, H, maxit + 1, beta, 1);
	/* 5.2. Compute x += Q[0:n, 0:iter] * y */
	cublasDgemv(cublas_handle, CUBLAS_OP_N,
							n, iter + 1, &one, Q, n,
							beta, 1, &one, x, 1);

#undef QCOL
#undef HCOL

#undef GVCOS
#undef GVSIN

	// cusparseDestroyDnVec(vec_r);
	// cusparseDestroyDnVec(vec_x);
	// cusparseDestroyDnVec(vec_tmp);

	CdamFreeDevice(Q, n * sizeof(f64) * (maxit + 1));
	CdamFreeDevice(H, (maxit + 1) * maxit * sizeof(f64));
	CdamFreeDevice(tmp, n * sizeof(f64));
	CdamFreeDevice(gv, 2 * maxit * sizeof(f64));
	CdamFreeDevice(beta, (maxit + 1) * sizeof(f64));

	cublasDestroy(cublas_handle);
}

Krylov* KrylovCreateCG(u32 max_iter, f64 atol, f64 rtol) {
	Krylov* ksp = KryloveInitPrivate(max_iter, atol, rtol);
	ksp->ksp_solve = CGSolvePrivate;
	ksp->pc_apply = PCJacobiApply;
	return ksp;
}

Krylov* KrylovCreateGMRES(u32 max_iter, f64 atol, f64 rtol) {
	Krylov* ksp = KryloveInitPrivate(max_iter, atol, rtol);
	ksp->ksp_solve = GMRESSolvePrivate;
	ksp->pc_apply = PCJacobiApply;
	return ksp;
}

void KrylovDestroy(Krylov* ksp) {
	CdamFreeDevice(ksp->ksp_ctx, ksp->ksp_ctx_size);
	CdamFreeDevice(ksp->pc_ctx, ksp->pc_ctx_size);
	CdamFreeHost(ksp, sizeof(Krylov));
}

/*
static void KrylovSetUp(Krylov* ksp, Matrix* A, f64* vec) {
	if(ksp->ksp_ctx) {
		return;
	}
	size_t buffer_size = 0;
	u32 n = MatrixNumRow(A);
	cusparseDnVecDescr_t vec_x, vec_b;
	cusparseCreateDnVec(&vec_x, n, vec, CUDA_R_64F);
	cusparseCreateDnVec(&vec_b, n, vec, CUDA_R_64F);

	cusparseSpMatDescr_t mat_A = MatrixCSRDescr(A);

	const f64 one = 1.0;
	cusparseSpMV_bufferSize(ksp->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
													&one, mat_A, vec_x, &one, vec_b, CUDA_R_64F,
													CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
	ksp->ksp_ctx_size = buffer_size;
	ksp->ksp_ctx = CdamMallocDevice(buffer_size);

#if CUDA_VERSION >= 12040
	cusparseSpMV_preprocess(ksp->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
													&one, mat_A, vec_x, &zero, vec_b, CUDA_R_64F,
													CUSPARSE_SPMV_ALG_DEFAULT, &ksp->ksp_ctx);
#endif

	cusparseDestroyDnVec(vec_x);
	cusparseDestroyDnVec(vec_b);
}
*/
void KrylovSolve(Krylov* ksp, Matrix* A, f64* x, f64* b) {
	// KrylovSetUp(ksp, A, b);
	ksp->ksp_solve(A, x, b, ksp);
}

__END_DECLS__
