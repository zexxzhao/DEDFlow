#include <string.h>
#include <float.h>
#include "alloc.h"

#include "json.h"
#include "csr.h"
#include "layout.h"
#include "pc.h"
#include "krylov.h"

__BEGIN_DECLS__

void GMRESResidualUpdatePrivate(f64*, f64*);


/**************************************************/
static void WrappedCdamMallocPrivate(void** ptr, ptrdiff_t size, MemType mem_type) {
	*ptr = CdamMalloc(size, mem_type);
}

static void WrappedMatvecMultPrivate(value_type alpha, void* A, void* x, value_type beta, void* y) {
	CdamParMatMultAdd(alpha, A, x, beta, y);
}

static void SolveCGPrivate(void* ksp, void* A, void* x, void* b) {
	// KrylovSolve((Krylov*)ksp, (Matrix*)A, (value_type*)x, (value_type*)b);
}

static void SetupCGPrivate(void* ksp, void* A, void* config) {
	// KrylovSetUp((Krylov*)ksp, (Matrix*)A, NULL);
}

static void SolveFlexGMRESPrivate(void* ksp, void* A, void* x, void* b) {
	value_type minus_one = -1.0, one = 1.0, zero = 0.0;
	CdamKrylov* krylov = (CdamKrylov*)ksp;
	index_type max_it = CdamKrylovMaxIt(krylov), min_it = CdamKrylovMinIt(krylov);
	index_type iter = 0;
	value_type atol = CdamKrylovAtol(krylov);
	value_type rtol = CdamKrylovRtol(krylov);
	index_type n = CdamParMatNumRowAll(A);
	value_type rnrm_init = 0.0, rnrm = 0.0;
	b32 converged = FALSE;
	index_type ldh = CEIL_DIV(max_it + 1, 32) * 32;

	CdamPC* pc = (CdamPC*)krylov->pc;

	value_type* ctx = (value_type*)krylov->ctx;


	CdamMemset(ctx, 0, krylov->ctx_size, DEVICE_MEM);
	value_type* Q = ctx;
	value_type* H = Q + n * (max_it + 1);
	value_type* tmp = H + ldh * max_it;
	value_type* gv = tmp + n * 2;
	value_type* beta = gv + 2 * max_it;

	/* r = b - A * x */
	krylov->op->vec_copy(n, Q + n * 0, b);

	krylov->op->matvec(-1.0, A, x, 1.0, Q + n * 0);

	krylov->op->vec_dot(n, Q + n * 0, Q + n * 0, &rnrm_init);

	/* beta = ||r|| */
	krylov->op->vec_dot(n, Q + n * 0, Q + n * 0, &rnrm_init);
	rnrm_init = sqrt(rnrm_init);
	krylov->op->vec_set(1, &rnrm_init, beta);

	krylov->op->vec_scal(n, (value_type)1.0 / rnrm, Q + n * 0);

	iter = 0;
	fprintf(stdout, "%3d) abs = %6.4e (tol = %6.4e) rel = %6.4e (tol = %6.4e)\n",
					iter, rnrm_init, atol, 1.0, rtol);
	
	while (!converged && iter < max_it) {
		/* Q[:, iter + 1] = A * inv(P) * Q[:, iter] */
		krylov->op->pc_apply(pc, Q + n * iter, tmp);

		/* vec_r = A * tmp */
		krylov->op->matvec(1.0, A, tmp, 0.0, Q + n * (iter + 1));

		/* Arnoldi process */
		/* H[0:iter+1, iter] = Q[0:n, 0:iter+1].T * Q[0:n, iter+1] */
		dgemv(BLAS_T, n, iter + 1,
					one, Q, n,
					Q + n * (iter + 1), 1,
					zero, H + ldh * iter, 1);
		/* Q[0:n, iter+1] -= Q[0:n, 0:iter+1] * H[0:iter+1, iter] */
		dgemv(BLAS_N, n, iter + 1,
					minus_one, Q, n,
					H + ldh * iter, 1,
					one, Q + n * (iter + 1), 1);

		/* H[iter+1, iter] = || Q[0:n, iter+1] || */ 
		SetPointerModeDevice();
		krylov->op->vec_dot(n, Q + n * (iter + 1), Q + n * (iter + 1), H + ldh * iter + iter + 1);
		SetPointerModeHost();
		krylov->op->vec_get(1, H + ldh * iter + iter + 1, &rnrm);
		krylov->op->vec_scal(n, (value_type)1.0 / rnrm, Q + n * (iter + 1));
		/* Apply Givens rotation to H[0:iter+2, iter] */
		SetPointerModeDevice();
		for (index_type i = 0; i < iter; i++) {
			drot(1, 
					H + ldh * iter + i, 1,
					H + ldh * iter + i + 1, 1, 
					gv[2 * i], gv[2 * i + 1]);
		}
		drotg(H[ldh * iter + iter],
					H[ldh * iter + iter + 1],
					gv + 2 * iter, gv + 2 * iter + 1);
		GMRESResidualUpdatePrivate(beta + iter, gv + 2 * iter);
		SetPointerModeHost();

		/* Check convergence */
		if((iter + 1) % 20 == 0) {
			krylov->op->vec_get(1, beta + iter + 1, &rnrm);
			rnrm = fabs(rnrm);
			if (iter >= min_it) {
				converged = rnrm < atol || rnrm < (rnrm_init + 1e-16) * rtol;
			}
			fprintf(stdout, "%3d) abs = %6.4e (tol = %6.4e) rel = %6.4e (tol = %6.4e)\n",
							iter + 1, rnrm, atol, rnrm / (rnrm_init + DBL_EPSILON), rtol);
			fflush(stdout);
		}
		iter++;
		CdamKrylovIter(krylov) = iter;
	}
	if(iter) { /* If the iteration is not zero */
		/* Get the solution */
		/* Solve upper triangular system H[0:iter, 0:iter] @ y = beta[0:iter+1] and beta <= y */
		dtrsv(BLAS_UP, BLAS_N, BLAS_NU,
					iter, H, ldh, beta, 1);
		/* Compute tmp = Q[0:n, 0:iter] * y */
		dgemv(BLAS_N, n, iter,
					one, Q, n, beta, 1,
					zero, tmp, 1);
		/* Apply preconditioner */
		krylov->op->pc_apply(pc, tmp, tmp + n);
		/* x += tmp */
		krylov->op->vec_axpy(n, 1.0, tmp + n, x);
	}
}
static void SetupFlexGMRESPrivate(void* ksp, void* A, void* config) {
	CdamKrylov* krylov = (CdamKrylov*)ksp;
	index_type max_it = CdamKrylovMaxIt(krylov);
	index_type ldh = CEIL_DIV(max_it + 1, 32) * 32;
	index_type n = CdamParMatNumRowAll(A);
	index_type buff_size = 0;

	buff_size += n * (max_it + 1);
	buff_size += ldh * max_it;
	buff_size += n * 2;
	buff_size += 2 * max_it;
	buff_size += max_it + 1;
	krylov->ctx = CdamTMalloc(value_type, buff_size, DEVICE_MEM);
	krylov->ctx_size = buff_size * sizeof(value_type);
}

static void VecCopyPrivate(index_type n, void* x, void* y) {
	CdamMemcpy(y, x, n * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
}

static void VecAxpyPrivate(index_type n, value_type alpha, void* x, void* y) {
	daxpy(n, alpha, x, 1, y, 1);
}

static void VecScalPrivate(index_type n, value_type alpha, void* x) {
	dscal(n, alpha, x, 1);
}

static void VecDotPrivate(index_type n, void* x, void* y, value_type* result) {
	ddot(n, x, 1, y, 1, result);
	MPI_Allreduce(MPI_IN_PLACE, result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

static void VecSetPrivate(index_type n, void* alpha, void* x) {
	CdamMemcpy(x, alpha, n * sizeof(value_type), DEVICE_MEM, HOST_MEM);
}

static void VecGetPrivate(index_type n, void* x, void* alpha) {
	CdamMemcpy(alpha, x, n * sizeof(value_type), HOST_MEM, DEVICE_MEM);
}

static void PCApplyPrivate(void* pc, void* x, void* y) {
	CdamPCApply((CdamPC*)pc, (value_type*)x, (value_type*)y);
}

void CdamKrylovCreate(CdamKrylov** ksp) {
	*ksp = CdamTMalloc(CdamKrylov, 1, HOST_MEM);
	CdamMemset(*ksp, 0, sizeof(CdamKrylov), HOST_MEM);
}

void CdamKrylovCreateCG(CdamKrylov** ksp) {
	*ksp = CdamTMalloc(CdamKrylov, 1, HOST_MEM);
	CdamMemset(*ksp, 0, sizeof(CdamKrylov), HOST_MEM);

	CdamKrylovMinIt(*ksp) = 20;
	CdamKrylovMaxIt(*ksp) = 100;
	CdamKrylovRtol(*ksp) = 1e-6;
	CdamKrylovAtol(*ksp) = 1e-6;

	(*ksp)->op->alloc = WrappedCdamMallocPrivate;
	(*ksp)->op->free = CdamFree;
	(*ksp)->op->matvec = WrappedMatvecMultPrivate;
	(*ksp)->op->solve = SolveCGPrivate;
	(*ksp)->op->setup = SetupCGPrivate;
	(*ksp)->op->vec_copy = VecCopyPrivate;
	(*ksp)->op->vec_axpy = VecAxpyPrivate;
	(*ksp)->op->vec_scal = VecScalPrivate;
	(*ksp)->op->vec_dot = VecDotPrivate;
	(*ksp)->op->vec_set = VecSetPrivate;
	(*ksp)->op->vec_get = VecGetPrivate;
	(*ksp)->op->pc_apply = PCApplyPrivate;
}

void CdamKrylovCreateFlexGMRES(CdamKrylov** ksp) {
	*ksp = CdamTMalloc(CdamKrylov, 1, HOST_MEM);
	CdamMemset(*ksp, 0, sizeof(CdamKrylov), HOST_MEM);

	CdamKrylovMinIt(*ksp) = 20;
	CdamKrylovMaxIt(*ksp) = 100;
	CdamKrylovRtol(*ksp) = 1e-6;
	CdamKrylovAtol(*ksp) = 1e-6;

	(*ksp)->op->alloc = WrappedCdamMallocPrivate;
	(*ksp)->op->free = CdamFree;
	(*ksp)->op->matvec = WrappedMatvecMultPrivate;
	(*ksp)->op->solve = SolveFlexGMRESPrivate;
	(*ksp)->op->setup = SetupFlexGMRESPrivate;
	(*ksp)->op->vec_copy = VecCopyPrivate;
	(*ksp)->op->vec_axpy = VecAxpyPrivate;
	(*ksp)->op->vec_scal = VecScalPrivate;
	(*ksp)->op->vec_dot = VecDotPrivate;
	(*ksp)->op->vec_set = VecSetPrivate;
	(*ksp)->op->vec_get = VecGetPrivate;
	(*ksp)->op->pc_apply = PCApplyPrivate;
}

void CdamKrylovDestroy(CdamKrylov* ksp) {
	CdamFree(ksp->ctx, ksp->ctx_size, DEVICE_MEM);
	CdamFree(ksp, sizeof(CdamKrylov), HOST_MEM);
}

void CdamKrylovSetup(CdamKrylov* ksp, void* A, void* config) {
	/* Pick the linear solver */
	ksp->op->alloc = WrappedCdamMallocPrivate;
	ksp->op->free = CdamFree;
	ksp->op->matvec = WrappedMatvecMultPrivate;
	ksp->op->vec_copy = VecCopyPrivate;
	ksp->op->vec_axpy = VecAxpyPrivate;
	ksp->op->vec_scal = VecScalPrivate;
	ksp->op->vec_dot = VecDotPrivate;
	ksp->op->vec_set = VecSetPrivate;
	ksp->op->vec_get = VecGetPrivate;

	if(strncmp(JSONGetItem(config, "Type")->valuestring, "CG", sizeof("CG")) == 0) {
		ksp->op->solve = SolveCGPrivate;
		ksp->op->setup = SetupCGPrivate;
	}
	else if(strncmp(JSONGetItem(config, "Type")->valuestring, "FGMRES", sizeof("FGMRES")) == 0) {
		ksp->op->solve = SolveFlexGMRESPrivate;
		ksp->op->setup = SetupFlexGMRESPrivate;
	}
	else {
		fprintf(stderr, "Unknown linear solver\n");
	}

	CdamKrylovMinIt(ksp) = 20;
	CdamKrylovMaxIt(ksp) = JSONGetItem(config, "MaxIteration")->valueint;
	CdamKrylovRtol(ksp) = JSONGetItem(config, "ResidualTolerance")->valuedouble;
	CdamKrylovAtol(ksp) = JSONGetItem(config, "AbsoluteTolerance")->valuedouble;

	ksp->op->setup(ksp, A, JSONGetItem(config, "Preconditioner"));


	CdamPCSetup((CdamPC*)ksp->pc, JSONGetItem(config, "Preconditioner"));
}

void CdamKrylovSolve(CdamKrylov* ksp, void* A, void* x, void* b) {
	ksp->op->solve(ksp, A, x, b);
}


__END_DECLS__
