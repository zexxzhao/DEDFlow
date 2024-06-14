#include <string.h>
#include <float.h>
#include <cublas_v2.h>
#include "alloc.h"

#include "csr.h"
#include "vec.h"
#include "matrix.h"
#include "krylov.h"

__BEGIN_DECLS__

static Krylov* KryloveInitPrivate(index_type max_iter, f64 atol, f64 rtol) {
	Krylov* ksp = (Krylov*)CdamMallocHost(SIZE_OF(Krylov));
	memset(ksp, 0, SIZE_OF(Krylov));
	ksp->max_iter = max_iter;
	ksp->atol = atol;
	ksp->rtol = rtol;

	return ksp;
}

static void PCSetUpPrivate(Matrix* A, void* ctx) {
	MatrixFS* mat = (MatrixFS*)A->data;
	const CSRAttr* spy = mat->spy1x1;
	index_type num_node = spy->num_row;

	Krylov* ksp = (Krylov*)ctx;

	ksp->pc_ctx_size = num_node * (3 * 3 + 3) * SIZE_OF(value_type);
	ksp->pc_ctx = CdamMallocDevice(ksp->pc_ctx_size);
}

void MatrixGetDiagBlock(const value_type* matval,
												index_type block_size,
												index_type num_row, index_type num_col,
												const index_type* row_ptr, const index_type* col_ind,
												value_type* diag, int lda, int stride);

static void PCGeneratePrivate(Matrix* A, void* ctx) {
	MatrixFS* mat = (MatrixFS*)A->data;
	index_type n_offset = mat->n_offset;
	const CSRAttr* spy = mat->spy1x1;
	index_type num_node = spy->num_row;

	Krylov* ksp = (Krylov*)ctx;
	value_type* pc_ctx = (value_type*)ksp->pc_ctx;
	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	CUGUARD(cudaGetLastError());
	/* Get all 3x3 diagonal blocks of mat[0] */
	CUGUARD(cudaGetLastError());
	MatrixGetDiagBlock(((MatrixCSR*)mat->mat[0]->data)->val, 3,
										 spy->num_row, spy->num_col, spy->row_ptr, spy->col_ind,
										 pc_ctx, 3, 3 * 3);
	cudaStreamSynchronize(0);
	CUGUARD(cudaGetLastError());
	/* Inverse the 3x3 diagonal matrices */
	value_type* inv = (value_type*)CdamMallocDevice(num_node * 3 * 3 * SIZE_OF(value_type));
	value_type** input_batch = (value_type**)CdamMallocDevice(num_node * SIZE_OF(value_type*) * 2);
	value_type** output_batch = input_batch + num_node;
	CUGUARD(cudaGetLastError());
	CUGUARD(cudaGetLastError());
	int* info = (int*)CdamMallocDevice(num_node * SIZE_OF(int) * 4);
	int* pivot = info + num_node;
	CUGUARD(cudaGetLastError());

	value_type** h_batch = (value_type**)CdamMallocHost(num_node * SIZE_OF(value_type*));
	for(int i = 0; i < num_node; i++) {
		h_batch[i] = pc_ctx + i * 3 * 3;
	}
	cudaMemcpy(input_batch, h_batch, num_node * SIZE_OF(value_type*), cudaMemcpyHostToDevice);
	for(int i = 0; i < num_node; i++) {
		h_batch[i] = inv + i * 3 * 3;
	}
	cudaMemcpy(output_batch, h_batch, num_node * SIZE_OF(value_type*), cudaMemcpyHostToDevice);
	CUGUARD(cudaGetLastError());
	cublasDgetrfBatched(cublas_handle, 3, input_batch, 3, pivot, info, num_node);
	cublasDgetriBatched(cublas_handle, 3, (const value_type* const*)input_batch, 3, pivot, output_batch, 3, info, num_node);
	cudaMemcpy(pc_ctx, inv, num_node * 3 * 3 * SIZE_OF(value_type), cudaMemcpyDeviceToDevice);


	/* mat[1][1] */
	MatrixGetDiag(mat->mat[1 * n_offset + 1], pc_ctx + (3 * 3 + 0) * num_node); 
	/* mat[2][2] */
	MatrixGetDiag(mat->mat[2 * n_offset + 2], pc_ctx + (3 * 3 + 1) * num_node);
	/* mat[3][3] */
	MatrixGetDiag(mat->mat[3 * n_offset + 3], pc_ctx + (3 * 3 + 2) * num_node);

	CdamFreeDevice(inv, num_node * 3 * 3 * SIZE_OF(value_type));
	CdamFreeDevice(input_batch, num_node * SIZE_OF(value_type*) * 2);
	CdamFreeDevice(info, num_node * SIZE_OF(int) * 4);
	CdamFreeHost(h_batch, num_node * SIZE_OF(value_type*));
	cublasDestroy(cublas_handle);
}

static void PCJacobiApply(Matrix* A, f64* x, f64* y, void* ctx) {
	MatrixFS* mat = (MatrixFS*)A->data;
	const CSRAttr* spy = mat->spy1x1;
	index_type num_node = spy->num_row;

	Krylov* ksp = (Krylov*)((void**)ctx)[0];
	cublasHandle_t cublas_handle = (cublasHandle_t)((void**)ctx)[1];

	value_type* pc_ctx = (value_type*)ksp->pc_ctx;

	f64 one = 1.0, zero = 0.0;

	cublasDgemvStridedBatched(cublas_handle,
														CUBLAS_OP_N,
														3, 3,
														&one,
														pc_ctx, 3, 3 * 3,
														x, 1, 3,
														&zero,
														y, 1, 3,
														num_node);


	VecPointwiseDiv(x + num_node * 3,
									pc_ctx + num_node * 3 * 3,
									y + num_node * 3,
									num_node * 3);

}

static void CGSolvePrivate(Matrix* A, f64* x, f64* b, void* ctx) {
	index_type n = MatrixNumRow(A);
	cusparseDnVecDescr_t vec_x, vec_b;
	cusparseCreateDnVec(&vec_x, n, x, CUDA_R_64F);
	cusparseCreateDnVec(&vec_b, n, b, CUDA_R_64F);

	// cusparseSpMatDescr_t mat_A = CSRMatrixDescr(A);
	// UNUSED(mat_A);
	/* TODO: Implement CG solver */
}

void GMRESResidualUpdatePrivate(f64*, f64*);

static f64 l2norm(f64* x, index_type n) {
	f64 norm;
	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	cublasDnrm2(cublas_handle, n, x, 1, &norm);

	cublasDestroy(cublas_handle);

	return norm;
}

/* GMRES solver */
static void GMRESSolvePrivate(Matrix* A, f64* x, f64* b, void* ctx) {
	f64 minus_one = -1.0, one = 1.0, zero = 0.0;
	Krylov* ksp = (Krylov*)ctx;
	index_type maxit = ksp->max_iter;
	f64 atol = ksp->atol;
	f64 rtol = ksp->rtol;
	index_type iter;
	index_type n = MatrixNumRow(A);
	f64 rnrm_init, rnrm;
	b32 converged;


	// cusparseHandle_t cusparse_handle = ksp->handle;
	// cusparseDnVecDescr_t vec_r, vec_x, vec_tmp;
	// cusparseSpMatDescr_t mat_A = CSRMatrixDescr(A);

	cublasHandle_t cublas_handle;
	// cublasStatus_t status;

#define QCOL(col) (Q + (col) * n)
#define HCOL(col) (H + (col) * (maxit + 1))
#define GVCOS(i) (gv + 2 * (i))
#define GVSIN(i) (gv + 2 * (i) + 1)
	CUGUARD(cudaGetLastError());

	/* Allocate memory for Q[n, maxit+1] */
	f64* Q = (f64*)CdamMallocDevice(n * SIZE_OF(f64) * (maxit + 1));
	CUGUARD(cudaGetLastError());
	cudaMemset(Q, 0, n * SIZE_OF(f64) * (maxit + 1));
	CUGUARD(cudaGetLastError());
	/* Allocate memeory for Hessenberg matrix H[(maxit + 1) * maxit] */
	f64* H = (f64*)CdamMallocDevice((maxit + 1) * maxit * SIZE_OF(f64));
	cudaMemset(H, 0, (maxit + 1) * maxit * SIZE_OF(f64));
	CUGUARD(cudaGetLastError());
	/* Allocate memory for tmp[n] */
	f64* tmp = (f64*)CdamMallocDevice(n * 2 * SIZE_OF(f64));
	cudaMemset(tmp, 0, n * 2 * SIZE_OF(f64));
	CUGUARD(cudaGetLastError());
	/* Allocate memory for Givens rotation */
	/* gv[::2] = cs, gv[1::2] = sn */
	f64* gv = (f64*)CdamMallocDevice(2 * maxit * SIZE_OF(f64));
	cudaMemset(gv, 0, 2 * maxit * SIZE_OF(f64));
	CUGUARD(cudaGetLastError());
	/* Allocate memory for residual vector */
	f64* beta = (f64*)CdamMallocDevice((maxit + 1) * SIZE_OF(f64));
	cudaMemset(beta, 0, (maxit + 1) * SIZE_OF(f64));
	CUGUARD(cudaGetLastError());

	/* Allocate memory for cublas handle */
	cublasCreate(&cublas_handle);
	CUGUARD(cudaGetLastError());

	void* pc_ctx[2] = {ksp, cublas_handle};

	/* 0. r = b - A * x */
	/* 0.0. r = b */
	cublasDcopy(cublas_handle, n, b, 1, QCOL(0), 1);
	/* 0.1. r -= A * x */
	MatrixAMVPBY(A, -1.0, x, 1.0, QCOL(0));
	CUGUARD(cudaGetLastError());


	converged = FALSE;
	cublasDnrm2(cublas_handle, n, QCOL(0), 1, &rnrm_init);
	CUGUARD(cudaGetLastError());

	/* Initialize beta[0] = rnrm_init */
	cublasSetVector(1, SIZE_OF(f64), &rnrm_init, 1, beta, 1);
	CUGUARD(cudaGetLastError());

	/* 1. Generate initial vector Q[:, 0] */
	/* Normalize Q[:, 0] */
	rnrm = (f64)1.0 / rnrm_init;
	cublasDscal(cublas_handle, n, &rnrm, QCOL(0), 1);

	CUGUARD(cudaGetLastError());

	iter = 0;

	fprintf(stdout, "%3d) abs = %6.4e (tol = %6.4e) rel = %6.4e (tol = %6.4e)\n",
					iter, rnrm_init, atol, 1.0, rtol);
	while (!converged && iter < maxit) {
		/* 2. Q[:, iter + 1] = A * inv(P) * Q[:, iter] */
		/* 2.0. Apply preconditioner: tmp[:] = inv(P) Q[:, iter] */
		ksp->pc_apply(A, QCOL(iter), tmp, pc_ctx);
		CUGUARD(cudaGetLastError());

		// ASSERT(!isnan(l2norm(tmp, n)));

		/* 2.1 let vec_r -> Q[:, iter + 1] */
		/* 2.2. vec_r = A * tmp */
		MatrixMatVec(A, tmp, QCOL(iter + 1));
		CUGUARD(cudaGetLastError());

		// ASSERT(!isnan(l2norm(QCOL(iter + 1), n)));

		/* 3. Arnoldi process */
		/* 3.1. H[0:iter+1, iter] = Q[0:n, 0:iter+1].T * Q[0:n, iter+1] */
		cublasDgemv(cublas_handle,
								CUBLAS_OP_T,
								n, iter + 1,
								&one,
								Q, n,
								QCOL(iter + 1), 1,
								&zero, 
								HCOL(iter), 1);
		CUGUARD(cudaGetLastError());

		// ASSERT(!isnan(l2norm(HCOL(iter), iter + 1)));

		/* 3.2. Q[0:n, iter+1] -= Q[0:n, 0:iter+1] * H[0:iter+1, iter] */
		cublasDgemv(cublas_handle,
								CUBLAS_OP_N,
							  n, iter + 1,
								&minus_one,
								Q, n,
								HCOL(iter), 1,
								&one,
								QCOL(iter + 1), 1);
		CUGUARD(cudaGetLastError());

		// ASSERT(!isnan(l2norm(QCOL(iter + 1), n)));

		/* 3.3. H[iter+1, iter] = || Q[0:n, iter+1] || */
		cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
		cublasDnrm2(cublas_handle, n, QCOL(iter + 1), 1, HCOL(iter) + iter + 1);
		cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
		CUGUARD(cudaGetLastError());
		
		/* 3.4. Q[0:n, iter+1] /= H[iter+1, iter] */
		cublasGetVector(1, SIZE_OF(f64), HCOL(iter) + iter + 1, 1, &rnrm, 1);
		rnrm = (f64)1.0 / rnrm;
		cublasDscal(cublas_handle, n, &rnrm, QCOL(iter + 1), 1);

		/* 4. Apply Givens rotation to H[0:iter+2, iter] */
		cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
		/* 4.0. Apply Givens rotation to H[0:iter+1, iter] using cs[0:iter+1] and sn[0:iter+1] */
		for (index_type i = 0; i < iter; i++) {
			cublasDrot(cublas_handle, 1,
								 HCOL(iter) + i, 1,
								 HCOL(iter) + i + 1, 1,
								 GVCOS(i), GVSIN(i));
		}

		/* 4.1. Compute Givens rotation */
		cublasDrotg(cublas_handle, HCOL(iter) + iter, HCOL(iter) + iter + 1, GVCOS(iter), GVSIN(iter));
		cudaMemset(HCOL(iter) + iter + 1, 0, SIZE_OF(f64));

		/* 4.2. Apply Givens rotation to H[iter:iter+2, iter] */
		// cublasDrot(cublas_handle, 1,
		// 					 HCOL(iter) + iter, 1,
		// 					 HCOL(iter) + iter + 1, 1,
		// 					 GVCOS(iter), GVSIN(iter));

		/* 4.3. beta[iter+1] = -sn[iter] * beta[iter]
		 *			beta[iter+0] =  cn[iter] * beta[iter] */
		GMRESResidualUpdatePrivate(beta+iter, GVCOS(iter));
		cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
		
		/* 4. Check convergence */
		cublasGetVector(1, SIZE_OF(f64), beta + iter + 1, 1, &rnrm, 1);
		rnrm = fabs(rnrm);
		if((iter + 1) % 5 == 0) {
			fprintf(stdout, "%3d) abs = %6.4e (tol = %6.4e) rel = %6.4e (tol = %6.4e)\n",
							iter + 1, rnrm, atol, rnrm / (rnrm_init + DBL_EPSILON), rtol);
		}
		if (rnrm < atol || rnrm < (rnrm_init + 1e-16) * rtol) {
			converged = TRUE;
		}
		iter++;
	}

	if(iter) { /* If the iteration is not zero */
		/* 5. Get the solution */
		/* 5.1. Solve upper triangular system H[0:iter, 0:iter] @ y = beta[0:iter+1] and beta <= y */
		cublasDtrsv(cublas_handle,
								CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
								iter,
								H, maxit + 1,
								beta, 1);
			
		/* 5.2. Compute tmp = Q[0:n, 0:iter] * y */
		cublasDgemv(cublas_handle,
								CUBLAS_OP_N,
								n, iter,
								&one,
								Q, n,
								beta, 1,
								&zero,
								tmp, 1);

		/* 5.3 Apply preconditioner */
		ksp->pc_apply(A, tmp, tmp + n, pc_ctx);

		/* 5.4. x += tmp */
		cublasDaxpy(cublas_handle, n, &one, tmp + n, 1, x, 1);
	}

#undef QCOL
#undef HCOL

#undef GVCOS
#undef GVSIN

	CdamFreeDevice(Q, n * SIZE_OF(f64) * (maxit + 1));
	CdamFreeDevice(H, (maxit + 1) * maxit * SIZE_OF(f64));
	CdamFreeDevice(tmp, n * SIZE_OF(f64));
	CdamFreeDevice(gv, 2 * maxit * SIZE_OF(f64));
	CdamFreeDevice(beta, (maxit + 1) * SIZE_OF(f64));

	cublasDestroy(cublas_handle);
}

Krylov* KrylovCreateCG(index_type max_iter, f64 atol, f64 rtol) {
	Krylov* ksp = KryloveInitPrivate(max_iter, atol, rtol);
	ksp->ksp_solve = CGSolvePrivate;
	ksp->pc_apply = PCJacobiApply;
	return ksp;
}

Krylov* KrylovCreateGMRES(index_type max_iter, f64 atol, f64 rtol) {
	Krylov* ksp = KryloveInitPrivate(max_iter, atol, rtol);
	ksp->ksp_solve = GMRESSolvePrivate;
	ksp->pc_apply = PCJacobiApply;
	return ksp;
}

void KrylovDestroy(Krylov* ksp) {
	CdamFreeDevice(ksp->ksp_ctx, ksp->ksp_ctx_size);
	CdamFreeDevice(ksp->pc_ctx, ksp->pc_ctx_size);
	CdamFreeHost(ksp, SIZE_OF(Krylov));
}

/*
static void KrylovSetUp(Krylov* ksp, Matrix* A, f64* vec) {
	if(ksp->ksp_ctx) {
		return;
	}
	size_t buffer_size = 0;
	index_type n = MatrixNumRow(A);
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
	if(ksp->pc_ctx == NULL) {
		PCSetUpPrivate(A, ksp);
	}
	PCGeneratePrivate(A, ksp);
	ksp->ksp_solve(A, x, b, ksp);
}

__END_DECLS__
