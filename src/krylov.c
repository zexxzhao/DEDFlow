#include <string.h>
#include <float.h>
#include "alloc.h"

#include "csr.h"
#include "vec.h"
#include "matrix.h"
#include "pc.h"
#include "krylov.h"

__BEGIN_DECLS__

static Krylov* KryloveInitPrivate(index_type max_iter, f64 atol, f64 rtol, void* handle) {
	Krylov* ksp = (Krylov*)CdamMallocHost(SIZE_OF(Krylov));
	memset(ksp, 0, SIZE_OF(Krylov));
	ksp->max_iter = max_iter;
	ksp->atol = atol;
	ksp->rtol = rtol;
	ksp->handle = handle;

	return ksp;
}

static void PCSetUpPrivate(Matrix* A, void* ctx) {
	// MatrixFS* mat = (MatrixFS*)A->data;
	// const CSRAttr* spy = mat->spy1x1;
	// index_type num_node = spy->num_row;

	// Krylov* ksp = (Krylov*)ctx;

	// ksp->pc_ctx_size = num_node * (3 * 3 + 3) * SIZE_OF(value_type);
	// ksp->pc_ctx = CdamMallocDevice(ksp->pc_ctx_size);
}

void MatrixGetDiagBlock(const value_type* matval,
												index_type block_size,
												index_type num_row, index_type num_col,
												const index_type* row_ptr, const index_type* col_ind,
												value_type* diag, int lda, int stride);

/*
static void PCGeneratePrivate(Matrix* A, void* ctx) {
	MatrixFS* mat = (MatrixFS*)A->data;
	index_type n_offset = mat->n_offset;
	const CSRAttr* spy = mat->spy1x1;
	index_type num_node = spy->num_row;

	Krylov* ksp = (Krylov*)ctx;
	value_type* pc_ctx = (value_type*)ksp->pc_ctx;
	cublasHandle_t cublas_handle = *(cublasHandle_t*)ksp->handle;

	CUGUARD(cudaGetLastError());
	CUGUARD(cudaGetLastError());
	MatrixGetDiagBlock(((MatrixCSR*)mat->mat[0]->data)->val, 3,
										 spy->num_row, spy->num_col, spy->row_ptr, spy->col_ind,
										 pc_ctx, 3, 3 * 3);
	// cudaStreamSynchronize(0);
	CUGUARD(cudaGetLastError());
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


	if(mat->mat[1 * n_offset + 1]) {
		MatrixGetDiag(mat->mat[1 * n_offset + 1], pc_ctx + (3 * 3 + 0) * num_node); 
	}
	else {
		SetValGPU(pc_ctx + (3 * 3 + 0) * num_node, num_node , 1.0);
	}
	if(mat->mat[2 * n_offset + 2]) {
		MatrixGetDiag(mat->mat[2 * n_offset + 2], pc_ctx + (3 * 3 + 1) * num_node);
	}
	else {
		SetValGPU(pc_ctx + (3 * 3 + 1) * num_node, num_node, 1.0);
	}
	if(mat->mat[3 * n_offset + 3]) {
		MatrixGetDiag(mat->mat[3 * n_offset + 3], pc_ctx + (3 * 3 + 2) * num_node);
	}
	else {
		SetValGPU(pc_ctx + (3 * 3 + 2) * num_node, num_node, 1.0);
	}

	CdamFreeDevice(inv, num_node * 3 * 3 * SIZE_OF(value_type));
	CdamFreeDevice(input_batch, num_node * SIZE_OF(value_type*) * 2);
	CdamFreeDevice(info, num_node * SIZE_OF(int) * 4);
	CdamFreeHost(h_batch, num_node * SIZE_OF(value_type*));

	if(0) {
		f64* h_pc_ctx = (f64*)CdamMallocHost(ksp->pc_ctx_size);
		cudaMemcpy(h_pc_ctx, pc_ctx, ksp->pc_ctx_size, cudaMemcpyDeviceToHost);
		FILE* fp = fopen("pc_ctx.txt", "w");
		for(int i = 0; i < num_node; i++) {
			fprintf(fp, "%.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e\n",
							h_pc_ctx[i * 3 * 3 + 0], h_pc_ctx[i * 3 * 3 + 1], h_pc_ctx[i * 3 * 3 + 2],
							h_pc_ctx[i * 3 * 3 + 3], h_pc_ctx[i * 3 * 3 + 4], h_pc_ctx[i * 3 * 3 + 5],
							h_pc_ctx[i * 3 * 3 + 6], h_pc_ctx[i * 3 * 3 + 7], h_pc_ctx[i * 3 * 3 + 8]);
		}
		fclose(fp);
		CdamFreeHost(h_pc_ctx, ksp->pc_ctx_size);
		exit(0);
	}
}
*/

/*
static void PCJacobiApply(Matrix* A, f64* x, f64* y, void* ctx) {
	// cudaMemcpy(y, x, MatrixNumRow(A) * SIZE_OF(f64), cudaMemcpyDeviceToDevice);
	// return;
	MatrixFS* mat = (MatrixFS*)A->data;
	const CSRAttr* spy = mat->spy1x1;
	index_type num_node = spy->num_row;

	Krylov* ksp = *(Krylov**)ctx;
	cublasHandle_t cublas_handle = *(cublasHandle_t*)ksp->handle;

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
									num_node * 1);

	cudaMemcpyAsync(y + num_node * 4, x + num_node * 4, num_node * 2 * SIZE_OF(f64), cudaMemcpyDeviceToDevice, 0);
	// cudaMemcpyAsync(y + num_node * 3, x + num_node * 3, 3 * num_node * SIZE_OF(f64), cudaMemcpyDeviceToDevice, 0);
}
*/

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

/* GMRES solver */
static void GMRESSolvePrivate(Matrix* A, f64* x, f64* b, void* ctx) {
	f64 minus_one = -1.0, one = 1.0, zero = 0.0;
	Krylov* ksp = (Krylov*)ctx;
	index_type maxit = ksp->max_iter;
	f64 atol = ksp->atol;
	f64 rtol = ksp->rtol;
	index_type iter;
	index_type n = MatrixNumRow(A);
	f64 rnrm_init = 0.0, rnrm = 0.0;
	b32 converged = FALSE;
	index_type ldh = CEIL_DIV(maxit + 1, 32) * 32;



	// cusparseHandle_t cusparse_handle = ksp->handle;
	// cusparseDnVecDescr_t vec_r, vec_x, vec_tmp;
	// cusparseSpMatDescr_t mat_A = CSRMatrixDescr(A);

	cublasHandle_t cublas_handle = *(cublasHandle_t*)ksp->handle;
	// cublasStatus_t status;

#define QCOL(col) (Q + (col) * n)
#define HCOL(col) (H + (col) * ldh)
#define GVCOS(i) (gv + 2 * (i))
#define GVSIN(i) (gv + 2 * (i) + 1)
	CUGUARD(cudaGetLastError());

	/* Allocate memory for Q[n, maxit+1] */
	f64* Q = (f64*)CdamMallocDevice(n * SIZE_OF(f64) * (maxit + 1));
	CUGUARD(cudaGetLastError());
	cudaMemset(Q, 0, n * SIZE_OF(f64) * (maxit + 1));
	CUGUARD(cudaGetLastError());
	/* Allocate memeory for Hessenberg matrix H[(maxit + 1) * maxit] */
	f64* H = (f64*)CdamMallocDevice(ldh * maxit * SIZE_OF(f64));
	cudaMemset(H, 0, ldh * maxit * SIZE_OF(f64));
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

	// cublasSetStream(cublas_handle, 0);

	// void* pc_ctx[2] = {ksp, cublas_handle};

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
		// ksp->pc_apply(A, QCOL(iter), tmp, pc_ctx);
		// ksp->pc_apply(A, QCOL(iter), tmp, &ksp);
		PCApply((PC*)ksp->pc, QCOL(iter), tmp);
		// cudaStreamSynchronize(0);
		CUGUARD(cudaGetLastError());

		// ASSERT(!isnan(l2norm(tmp, n)));

		/* 2.1 let vec_r -> Q[:, iter + 1] */
		/* 2.2. vec_r = A * tmp */
		// cudaMemcpy(QCOL(iter + 1), tmp, n * SIZE_OF(f64), cudaMemcpyDeviceToDevice);
		MatrixMatVec(A, tmp, QCOL(iter + 1));
		// cudaStreamSynchronize(0);

		// ASSERT(!isnan(l2norm(QCOL(iter + 1), n)));

		/* 3. Arnoldi process */

		// ASSERT(!isnan(l2norm(HCOL(iter), iter + 1)));

		if(1) {
		/* 3.1. H[0:iter+1, iter] = Q[0:n, 0:iter+1].T * Q[0:n, iter+1] */
			// cudaStreamSynchronize(0);
			cublasDgemv(cublas_handle,
									CUBLAS_OP_T,
									n, iter + 1,
									&one,
									Q, n,
									QCOL(iter + 1), 1,
									&zero, 
									HCOL(iter), 1);
		/* 3.2. Q[0:n, iter+1] -= Q[0:n, 0:iter+1] * H[0:iter+1, iter] */
			// cudaStreamSynchronize(0);
			cublasDgemv(cublas_handle,
									CUBLAS_OP_N,
									n, iter + 1,
									&minus_one,
									Q, n,
									HCOL(iter), 1,
									&one,
									QCOL(iter + 1), 1);

			// cudaStreamSynchronize(0);
		}
		if(0) {
			index_type begin = 0;
			index_type end = iter + 1;
			int stride = 64;
			for(; begin < end; begin += stride) {
				index_type ncol = (end - begin < stride ? end - begin : stride);
				cublasDgemv(cublas_handle,
										CUBLAS_OP_T,
										n, stride,
										&one,
										QCOL(begin), n,
										QCOL(iter + 1), 1,
										&zero, 
										HCOL(iter) + begin, 1);
				cublasDgemv(cublas_handle,
										CUBLAS_OP_N,
										n, ncol,
										&minus_one,
										QCOL(begin), n,
										HCOL(iter) + begin, 1,
										&one,
										QCOL(iter + 1), 1);
			}
		}
		if(0) {
			/* Modified Gram-Schmidt */
			for(index_type i = 0; i < iter + 1; i++) {
				cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
				cublasDdot(cublas_handle, n, QCOL(i), 1, QCOL(iter + 1), 1, HCOL(iter) + i);

				cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
				cublasDscal(cublas_handle, 1, &minus_one, HCOL(iter) + i, 1);

				cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
				cublasDaxpy(cublas_handle, n, HCOL(iter) + i, QCOL(i), 1, QCOL(iter + 1), 1);
			}
			cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
			cublasDscal(cublas_handle, ldh, &minus_one, HCOL(iter), 1);
		}

		if(1) {
			/* 3.3. H[iter+1, iter] = || Q[0:n, iter+1] || */
			cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
			cublasDnrm2(cublas_handle, n, QCOL(iter + 1), 1, HCOL(iter) + iter + 1);
			cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
			CUGUARD(cudaGetLastError());
			
			/* 3.4. Q[0:n, iter+1] /= H[iter+1, iter] */
			cublasGetVector(1, SIZE_OF(f64), HCOL(iter) + iter + 1, 1, &rnrm, 1);
			rnrm = (f64)1.0 / rnrm;
			cublasDscal(cublas_handle, n, &rnrm, QCOL(iter + 1), 1);
		}
		if(0) {
			rnrm = 0.0;
			cublasDdot(cublas_handle, iter + 1, HCOL(iter), 1, HCOL(iter), 1, &rnrm);
			fprintf(stdout, "|H| = %a\t", rnrm);
			cublasDdot(cublas_handle, n, QCOL(iter + 1), 1, QCOL(iter + 1), 1, &rnrm);
			fprintf(stdout, "rnrm = %a\t", rnrm);
			fflush(stdout);

			rnrm = sqrt(rnrm);
			cublasSetVector(1, SIZE_OF(f64), &rnrm, 1, HCOL(iter) + iter + 1, 1);
			rnrm = 1.0 / rnrm;
			cublasDscal(cublas_handle, n, &rnrm, QCOL(iter + 1), 1);
			fprintf(stdout, "\n");
			iter++;
			continue;
		}
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
		if((iter + 1) % 10 == 0) {
			cublasGetVector(1, SIZE_OF(f64), beta + iter + 1, 1, &rnrm, 1);
			rnrm = fabs(rnrm);
			fprintf(stdout, "%3d) abs = %6.4e (tol = %6.4e) rel = %6.4e (tol = %6.4e)\n",
							iter + 1, rnrm, atol, rnrm / (rnrm_init + DBL_EPSILON), rtol);
			fflush(stdout);
			if (rnrm < atol || rnrm < (rnrm_init + 1e-16) * rtol) {
				converged = TRUE;
			}
		}
		iter++;
	}

	if(iter) { /* If the iteration is not zero */
		/* 5. Get the solution */
		/* 5.1. Solve upper triangular system H[0:iter, 0:iter] @ y = beta[0:iter+1] and beta <= y */
		cublasDtrsv(cublas_handle,
								CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
								iter,
								H, ldh,
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
		// ksp->pc_apply(A, tmp, tmp + n, pc_ctx);
		// ksp->pc_apply(A, tmp, tmp + n, &ksp);
		PCApply((PC*)ksp->pc, tmp, tmp + n);

		/* 5.4. x += tmp */
		cublasDaxpy(cublas_handle, n, &one, tmp + n, 1, x, 1);
	}

#undef QCOL
#undef HCOL

#undef GVCOS
#undef GVSIN

	CdamFreeDevice(Q, n * SIZE_OF(f64) * (maxit + 1));
	CdamFreeDevice(H, ldh * maxit * SIZE_OF(f64));
	CdamFreeDevice(tmp, n * SIZE_OF(f64));
	CdamFreeDevice(gv, 2 * maxit * SIZE_OF(f64));
	CdamFreeDevice(beta, (maxit + 1) * SIZE_OF(f64));

}

Krylov* KrylovCreateCG(index_type max_iter, f64 atol, f64 rtol, void* handle) {
	Krylov* ksp = KryloveInitPrivate(max_iter, atol, rtol, handle);
	ksp->ksp_solve = CGSolvePrivate;
	// ksp->pc_apply = NULL;
	return ksp;
}

Krylov* KrylovCreateGMRES(index_type max_iter, f64 atol, f64 rtol, void* handle) {
	Krylov* ksp = KryloveInitPrivate(max_iter, atol, rtol, handle);
	ksp->ksp_solve = GMRESSolvePrivate;
	// ksp->pc_apply = NULL;
	return ksp;
}

void KrylovDestroy(Krylov* ksp) {
	PCDestroy(ksp->pc);
	CdamFreeDevice(ksp->ksp_ctx, ksp->ksp_ctx_size);
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
	PC* pc = (PC*)ksp->pc;
	if(pc == NULL || pc->mat != A) {
		char cfg_content[2048];
		sprintf(cfg_content, "AMGX.json");	
		/*
		sprintf(cfg_content, "config_version=2, \
													solver=AMG, \
													solver:preconditioner=NOSOLVER, \
													solver:cycle=V, \
													solver:smoother=BLOCK_JACOBI, \
													solver:coarse_solver=DENSE_LU_SOLVER, \
													solver:use_scalar_norm=2, \
													solver:max_iters=100, \
													solver:monitor_residual=1, \
													solver:convergence=RELATIVE_INI_CORE, \
													solver:tolerance=1e-16, \
													print_solve_stats=0, \
													print_grid_stats=0, \
													obtain_timings=0, \
													solver:norm=L2");
													*/
		/*
		sprintf(cfg_content, "config_version=2,"
												 "solver:preconditioner:error_scaling=0,"
												 "solver:preconditioner:print_grid_stats=1,"
												 "solver:preconditioner:max_uncolored_percentage=0.05,"
												 "solver:preconditioner:algorithm=AGGREGATION,"
												 "solver:preconditioner:solver=AMG,"
												 "solver:preconditioner:smoother=MULTICOLOR_DILU,"
												 "solver:preconditioner:presweeps=0,"
												 "solver:preconditioner:selector=SIZE_2,"
												 "solver:preconditioner:coarse_solver=DENSE_LU_SOLVER,"
												 "solver:preconditioner:max_iters=1,"
												 "solver:preconditioner:postsweeps=3,"
												 "solver:preconditioner:min_coarse_rows=32,"
												 "solver:preconditioner:relaxation_factor=0.75,"
												 "solver:preconditioner:scope=amg,"
												 "solver:preconditioner:max_levels=100,"
												 "solver:preconditioner:matrix_coloring_scheme=PARALLEL_GREEDY,"
												 "solver:preconditioner:cycle=V,"
												 "solver:use_scalar_norm=1,"
												 "solver:solver=FGMRES,"
												 "solver:print_solve_stats=1,"
												 "solver:obtain_timings=1,"
												 "solver:max_iters=100,"
												 "solver:monitor_residual=1,"
												 "solver:gmres_n_restart=10,"
												 "solver:convergence=RELATIVE_INI_CORE,"
												 "solver:scope=main,"
												 "solver:tolerance=1e-10,"
												 "solver:norm=L2");
												 */
		PCDestroy((PC*)pc);
		index_type n_sec = 4;
		const CSRAttr* spy = ((MatrixFS*)A->data)->spy1x1;
		Matrix* A00 = ((MatrixFS*)A->data)->mat[0 * 4 + 0];
		Matrix* A11 = ((MatrixFS*)A->data)->mat[1 * 4 + 1];
		index_type n = spy->num_row;
		index_type offset[] = {0 * n, 3 * n, 4 * n, 5 * n};
		ksp->pc = (void*)PCCreateDecomposition(A, n_sec, offset, ksp->handle);
		pc = (PC*)ksp->pc;
		((PCDecomposition*)pc->data)->pc[0] = PCCreateJacobi(A00, 3, ksp->handle);
		((PCDecomposition*)pc->data)->pc[1] = PCCreateJacobi(A11, 1, ksp->handle);
		// ((PCDecomposition*)pc->data)->pc[1] = PCCreateAMGX(A11, cfg_content);
		((PCDecomposition*)pc->data)->pc[2] = PCCreateNone(NULL, n);
		((PCDecomposition*)pc->data)->pc[3] = PCCreateNone(NULL, n);
	}
	PCSetup(pc);
	ksp->ksp_solve(A, x, b, ksp);
}

__END_DECLS__
