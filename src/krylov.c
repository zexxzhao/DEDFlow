#include "alloc.h"

#include "pc.cuh"
#include "krylov.h"

__BEGIN_DECLS__

static Krylov* KryloveInitPrivate(u32 max_iter, f64 atol, f64 rtol) {
	Krylov* ksp = (Krylov*)CdamMallocHost(sizeof(Krylov));
	ksp->max_iter = max_iter;
	ksp->atol = atol;
	ksp->rtol = rtol;

	ksp->handle = NULL;
	ksp->KSPSolve = NULL;
	ksp->PCApply = NULL;
	ksp->ctx = NULL;
	return ksp;
}
static void PCJacobiApply(CSRMatrix* A, f64* x, f64* b) {
	u32 n = CSRMatrixNumRows(A);
	u32 nnz = CSRMatrixNNZ(A);
	u32* row_ptr = CSRMatrixRowPtr(A);
	u32* col_idx = CSRMatrixColIdx(A);
	f64* val = CSRMatrixVal(A);

	i32 block_size = 1024;
	i32 num_blocks = (n + block_size - 1) / block_size;
	if(x == NULL or x == b) {
		PCJacobiApplyInPlaceKernel<<<num_blocks, block_size>>>(n, nnz, row_ptr, col_idx, val, b);
	}
	else {
		PCJacobiApplyKernel<<<num_blocks, block_size>>>(n, nnz, row_ptr, col_idx, val, b, x);
	}
	
}
static void CGSolvePrivate(CSRMatrix* A, f64* x, f64* b, u32 max_iter, f64 atol, f64 rtol, PCApplyFunc pc, void* ctx) {
	u32 n = CSRMatrixNumRows(A);
	cusparseDnVecDescr_t vec_x, vec_b;
	cusparseCreateDnVec(&vec_x, n, x);
	cusparseCreateDnVec(&vec_b, n, b);

	cusparseDnMatDescr_t mat_A = CSRMatrixDescr(A);
	/* TODO: Implement CG solver */
}

/* GMRES solver */
static void GMRESSolvePrivate(CSRMatrix* A, f64* x, f64* b, void* ctx) {
	const f64 minus_one = -1.0, one = 1.0, zero = 0.0;
	Krylov* ksp = (Krylov*)ctx;
	u32 maxit = ksp->max_iter;
	f64 atol = ksp->atol;
	f64 rtol = ksp->rtol;
	void* buffer_mv = ksp->ksp_ctx;
	u32 n = CSRMatrixNumRows(A);

	cusparseHandle_t cusparse_handle = ksp->handle;
	cusparseDnVecDescr_t vec_r, vec_x, vec_tmp;
	cusparseDnMatDescr_t mat_A = CSRMatrixDescr(A);

	cublasHandle_t cublas_handle;

	cublasCreate(&cublas_handle);

	/* Allocate memory for Q[maxit, n] */
	f64* Q = (f64*)CdamMallocDevice(n * sizeof(64) * maxit);
	cudaMemset(Q, 0, n * sizeof(f64) * maxit);
	/* Allocate memeory for Hessenberg matrix H[(maxit + 1) * maxit] */
	f64* H = (f64*)CdamMallocDevice((maxit + 1) * maxit * sizeof(f64));
	cudaMemset(H, 0, (maxit + 1) * maxit * sizeof(f64));
	/* Allocate memory for tmp[n] */
	f64* tmp = (f64*)CdamMallocDevice(n * sizeof(f64));
	cudaMemset(tmp, 0, n * sizeof(f64));
	/* Allocate memory for Givens rotation */
	f64* cs = (f64*)CdamMallocDevice(2 * maxit * sizeof(f64));
	cudaMemset(cs, 0, 2 * maxit * sizeof(f64));
	f64* sn = cs + maxit;
	/* Allocate memory for residual vector */
	f64* beta = (f64*)CdamMallocDevice((maxit + 1) * sizeof(f64));
	cudaMemset(beta, 0, (maxit + 1) * sizeof(f64));
	cublasSetVector(1, sizeof(f64), &one, 1, beta, 1);

	/* 0. r = b - A * x */
	cusparseCreateDnVec(&vec_x, n, x);
	cusparseCreateDnVec(&vec_r, n, Q + n * 0);
	cusparseCreateDnVec(&vec_tmp, n, tmp);
	/* 0.0. r = b */
	cublasDcopy(cublas_handle, n, b, 1, Q, 1);
	/* 0.1. r -= A * x */
	cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
							 &minus_one, mat_A, vec_x, &one, vec_r,
							 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer_mv);


	b32 converged = FALSE;
	f64 rnrm_init, rnrm;
	cublasDnrm2(handle, n, vec_r, &rnrm_init);
	if (rnrm_init < atol) {
		converged = TRUE;
	}

	/* Normalize Q[:, 0] */
	rnrm = (f64)1.0 / rnrm_init;
	cublasDscal(cublas_handle, n, &rnrm, Q, 1);


	u32 iter = 1;
	while (!converged && iter < maxit) {
		/* 1. Q[:, iter] = A * inv(P) * Q[:, iter-1] */
		ksp->pc_apply(A, Q + (iter - 1) * n, tmp);
		cusparseDestroyDnVec(vec_r);
		cusparseCreateDnVec(&vec_r, n, Q + iter * n);
		cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
								 &one, mat_A, vec_tmp, &zero, vec_r,
								 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer_mv);
		/* 2. Arnoldi process */
		/* 2.1. H[0:iter, iter-1] = Q[0:n, 0:iter].T * Q[0:n, iter] */
		cublasDgemv(cublas_handle, CUBLAS_OP_T, n, iter, &one, Q, n, Q + iter * n, 1, &zero, H + (iter-1) * maxit, 1);
		/* 2.2. Q[0:n, iter] -= Q[0:n, 0:iter] * H[0:iter, iter-1] */
		cublasDgemv(cublas_handle, CUBLAS_OP_N, n, iter, &minus_one, Q, n, H + (iter-1) * maxit, 1, &one, Q + iter * n, 1);
		/* 2.3. H[iter, iter-1] = ||Q[0:n, iter]|| */
		cublasDnrm2(cublas_handle, n, Q + iter * n, H + (iter-1) * maxit + iter);
		/* 2.4. Q[0:n, iter] /= H[iter, iter-1] */
		cublasGetVector(1, sizeof(f64), H + (iter-1) * maxit + iter, 1, &rnrm, 1);
		rnrm = (f64)1.0 / rnrm;
		cublasDscal(cublas_handle, n, &rnrm, Q + iter * n, 1);

		/* 3. Apply Givens rotation to H[0:iter+1, iter-1] */
		/* 3.0. Apply Givens rotation to H[0:iter, iter-1] using cs[0:iter] and sn[0:iter] */
		for (u32 i = 0; i < iter - 1; i++) {
			cublasDrot(cublas_handle, 1,
								 H + iter * maxit + i, 1,
								 H + iter * maxit + i + 1, 1,
								 cs + i, sn + i);
		}
		/* 3.1. Compute Givens rotation */
		cublasDrotg(cublas_handle,
								H + (iter-1) * maxit + iter - 1, H + iter * maxit + iter,
								cs + iter - 1, sn + iter - 1);
		/* 3.2. Apply Givens rotation to H[iter-1:iter+1, iter-1] */
		cublasDrot(cublas_handle, 1,
							 H + (iter-1) * maxit + iter - 1, 1,
							 H + iter * maxit + iter - 1, 1,
							 cs + iter - 1, sn + iter - 1);
		/* 3.3. beta[iter  ] = -sn[iter - 1] * beta[iter-1]
		 *			beta[iter-1] =  cn[iter - 1] * beta[iter-1] */
		cublas

		/* 4. Check convergence */
		cublasGetVector(1, sizeof(f64), beta + iter, 1, &rnrm, 1);
		if (rnrm < atol) {
			converged = TRUE;
		}
		else {
			rnrm = fabs(sn[iter - 1]) * rnrm;
			if (rnrm < rtol) {
				converged = TRUE;
			}
		}
	}


	cusparseDestroyDnVec(vec_r);
	cusparseDestroyDnVec(vec_x);
	cusparseDestroyDnVec(vec_tmp);

	CdamFreeDevice(Q, n * sizeof(f64) * maxit);
	CdamFreeDevice(H, (maxit + 1) * maxit * sizeof(f64));
	CdamFreeDevice(tmp, n * sizeof(f64));
	CdamFreeDevice(cs, 2 * maxit * sizeof(f64));
	CdamFreeDevice(beta, (maxit + 1) * sizeof(f64));

	cublasDestroy(cublas_handle);
}

Krylov* KrylovCreateCG(u32 max_iter, f64 atol, f64 rtol, void* ctx) {
	Krylov* ksp = KryloveInitPrivate(max_iter, atol, rtol);
	cusparseCreate(&ksp->handle);
	ksp->ksp_solve = CGSolvePrivate;
	ksp->pc_apply = PCJacobiApply;
}

Krylov* KryloveCreateGMRES(u32 max_iter, f64 atol, f64 rtol, void* ctx) {
	Krylov* ksp = KryloveInitPrivate(max_iter, atol, rtol);
	cusparseCreate(&ksp->handle);
	ksp->ksp_solve = GMRESSolvePrivate;
	ksp->pc_apply = PCJacobiApply;
}

void KrylovDestroy(Krylov* ksp) {
	cusparseDestroy(ksp->handle);
	CdamFreeDevice(ksp->ksp_ctx, ksp->ksp_ctx_size);
	CdamFreeDevice(ksp->pc_ctx, ksp->pc_ctx_size);
	CdamFreeHost(ksp, sizeof(Krylov));
}

static void KrylovSetUp(Krylov* ksp, CSRMatrix* A, f64* vec) {
	if(ksp->ksp_ctx) {
		return;
	}
	u32 buffer_size = 0;
	u32 n = CSRMatrixNumRows(A);
	cusparseDnVecDescr_t vec_x, vec_b;
	cusparseCreateDnVec(&vec_x, n, vec);
	cusparseCreateDnVec(&vec_b, n, vec);

	cusparseSpMatDescr_t mat_A = CSRMatrixDescr(A);

	const f64 one = 1.0;
	cusparseSpMV_bufferSize(ksp->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
													&one, mat_A, vec_x, &one, vec_b, CUDA_R_64F,
													CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
	ksp->ksp_ctx_size = buffer_size;
	ksp->ksp_ctx = CdamMallocDevice(buffer_size);

	/* Use cusparseSpMV_preprocess if CUDA VERSION is newer and equal than 12.4 */
#if CUDA_VERSION >= 12040
	cusparseSpMV_preprocess(ksp->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
													&one, mat_A, vec_x, &zero, vec_b, CUDA_R_64F,
													CUSPARSE_SPMV_ALG_DEFAULT, &ksp->ksp_ctx);
#endif

	cusparseDestroyDnVec(vec_x);
	cusparseDestroyDnVec(vec_b);
}

void KrylovSolve(Krylov* ksp, CSRMatrix* A, f64* x, f64* b) {
	KrylovSetUp(ksp, A, b);
	ksp->ksp_solve(A, x, b, ksp);
}

__END_DECLS__
