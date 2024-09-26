#include <mpi.h>
#include "json.h"
#include "alloc.h"

#include "Mesh.h"
#include "blas.h"
#include "layout.h"
#include "NewtonSolver.h"

void CdamKrylovSolve(CdamKrylov* solver, void* A, void* x, void* b); 


static b32 CheckConvergencePrivate(void* data, void* x, b32 verbose) {
	CdamNewtonSolver* solver = (CdamNewtonSolver*)data;
	CdamLayout* layout = (CdamLayout*)solver->layout;


	value_type atol = solver->atol;
	value_type rtol = solver->rtol;
	
	index_type i;
	index_type n = CdamLayoutNumComponent(layout);

	value_type* r = solver->rnorm;
	value_type* r0 = solver->rnorm_init;

	VecNorm(x, layout, NULL, r);


	if(solver->iter == 0) {
		memcpy(r0, r, n * sizeof(value_type));
	}

	b32 converged = 1;
	for(i = 0; i < n; i++) {
		converged = converged && (r[i] < atol || r[i] < rtol * r0[i]);
	}

	if(verbose) {
		for(i = 0; i < n; ++i) {
			fprintf(stdout, "Newton %d) abs=%.17e rel=%8.6e (tol=%8.6e)\n",
							solver->iter, r[i], r[i] / MAX(r0[i], 1e-16), rtol);
		}
	}
	return converged;
}

static void UpdateSolutionPrivate(void* data, void *sol) {
	CdamNewtonSolver* solver = (CdamNewtonSolver*)data;
	CdamLayout* layout = (CdamLayout*)solver->layout;
	value_type relaxation;
	value_type* dx = (value_type*)solver->dx;
	index_type n = CdamLayoutLen(layout);
	
	relaxation = solver->relaxation;
	if(relaxation == 0.0) {
		relaxation = -1.0;
	}
	if(relaxation > 0.0) {
		relaxation *= -1.0;
	}

	daxpy(n, relaxation, dx, 1, sol, 1);
	
}

void CdamNewtonSolverCreate(MPI_Comm comm, CdamNewtonSolver **solver) {
	*solver = (CdamNewtonSolver*)CdamMallocHost(sizeof(CdamNewtonSolver));
	memset(*solver, 0, sizeof(CdamNewtonSolver));
	(*solver)->comm = comm;
	(*solver)->maxit = 100;
	(*solver)->rtol = 1.0e-6;
	(*solver)->atol = 1.0e-12;
	(*solver)->relaxation = 1.0;

	(*solver)->op->is_converged = CheckConvergencePrivate;
	(*solver)->op->update_solution = UpdateSolutionPrivate;
}

void CdamNewtonSolverDestroy(CdamNewtonSolver *solver) {
	CdamFreeHost(solver, sizeof(CdamNewtonSolver));
	solver = NULL;
}

void CdamNewtonSolverConfig(CdamNewtonSolver *solver, void* config) {
	cJSON *json = (cJSON*)config;
	solver->rtol = JSONGetItem(json, "RelativeTolerance")->valuedouble;
	solver->atol = JSONGetItem(json, "AbsoluteTolerance")->valuedouble;
	solver->maxit = JSONGetItem(json, "MaxIterations")->valueint;
	solver->relaxation = JSONGetItem(json, "RelaxationFactor")->valuedouble;

}


void CdamNewtonSolverSetRelativeTolerance(CdamNewtonSolver *solver, value_type rtol) {
	solver->rtol = rtol;
}

void CdamNewtonSolverSetAbsoluteTolerance(CdamNewtonSolver *solver, value_type atol) {
	solver->atol = atol;
}

void CdamNewtonSolverSetMaxIterations(CdamNewtonSolver *solver, int maxit) {
	solver->maxit = maxit;
}

void CdamNewtonSolverSetLinearForm(CdamNewtonSolver *solver, NewtonSolverFunctionLinear linear_form) {
	solver->op->linear_form = linear_form;
}

void CdamNewtonSolverSetBilinearForm(CdamNewtonSolver *solver, NewtonSolverFunctionBilinear bilinear_form) {
	solver->op->bilinear_form = bilinear_form;
}

void CdamNewtonSolverSetKrylov(CdamNewtonSolver *solver, CdamKrylov *lsolver) {
	solver->lsolver = lsolver;
}

void CdamNewtonSolverSetLinearSystem(CdamNewtonSolver *solver, void* A, void* dx, void* b) {
	solver->A = A;
	solver->dx = dx;
	solver->b = b;
}


void CdamNewtonSolverGetRelativeTolerance(CdamNewtonSolver *solver, value_type *rtol) {
	if(rtol)
		*rtol = solver->rtol;
}

void CdamNewtonSolverGetAbsoluteTolerance(CdamNewtonSolver *solver, value_type *atol) {
	if(atol)
		*atol = solver->atol;
}

void CdamNewtonSolverGetMaxIterations(CdamNewtonSolver *solver, int *maxit) {
	if(maxit)
		*maxit = solver->maxit;
}

void CdamNewtonSolverGetLinearSolver(CdamNewtonSolver *solver, CdamKrylov **lsolver) {
	if(lsolver)
		*lsolver = solver->lsolver;
}
// void CdamNewtonSolverGetLinearSystem(CdamNewtonSolver *solver, void** A, void** dx, void** b) {
// 	if(A) *A = solver->A;
// 	if(dx) *dx = solver->dx;
// 	if(b) *b = solver->b;
// }


void CdamNewtonSolverSolve(CdamNewtonSolver* solver, value_type* x) {
	CdamKrylov* lsolver = solver->lsolver;
	void* A = solver->A;
	void* b = solver->b;
	void* dx = solver->dx;

	index_type maxit = solver->maxit;

	b32 converged = 0;


	
	/* Construct the right-hand side */
	solver->op->linear_form(solver, b, NULL);


	solver->iter = 0;
	converged = solver->op->is_converged(solver, b, 1);

	while(! converged || solver->iter < maxit) {
		/* Construct the Jacobian matrix */
		solver->op->bilinear_form(solver, A, NULL);

		/* Solve the linear system */
		CdamKrylovSolve(lsolver, A, dx, b);
		
		/* Update the solution */
		solver->op->update_solution(solver, x);

		/* Construct the right-hand side */
		solver->op->linear_form(solver, b, NULL);

		converged = solver->op->is_converged(solver, b, 1);
		if(converged) {
			break;
		}
		solver->iter++;
	}

}

// static
// void SolveFlowSystem(Mesh3D* mesh,
// 										 f64* wgold, f64* dwgold, f64* dwg,
// 										 Matrix* J, f64* F, f64* dx,
// 										 Krylov* ksp,
// 										 Dirichlet** bcs, index_type nbc) {
// 
// #ifdef DBG_TET
// 	index_type maxit = 1;
// #else
// 	index_type maxit = 4;
// #endif
// 	index_type iter = 0;
// 	f64 tol = 0.5e-3;
// 	b32 converged = FALSE;
// 	f64 rnorm[4], rnorm_init[4];
// 	f64 minus_one = -1.0;
// 	clock_t start, end;
// 
// 	f64 fact1[] = {1.0 - kALPHAM, kALPHAM, kALPHAM - 1.0, -kALPHAM};
// 	f64 fact2[] = {kDT * kALPHAF * (1.0 - kGAMMA), kDT * kALPHAF * kGAMMA,
// 								 -kDT * kALPHAF * (1.0 - kGAMMA), -kDT * kALPHAF * kGAMMA};
// 
// 
// 	index_type num_node = Mesh3DDataNumNode(Mesh3DHost(mesh));
// 
// 	f64* wgalpha = (f64*)CdamMallocDevice(num_node * SIZE_OF(f64) * BS);
// 	f64* dwgalpha = (f64*)CdamMallocDevice(num_node * SIZE_OF(f64) * BS);
// 
// 
// 	/* dwgalpha = fact1[0] * dwgold + fact1[1] * dwg */ 
// 	cudaMemset(dwgalpha, 0, num_node * SIZE_OF(f64) * BS);
// 	BLAS_CALL(Daxpy, num_node * BS, fact1 + 0, dwgold, 1, dwgalpha, 1);
// 	BLAS_CALL(Daxpy, num_node * BS, fact1 + 1, dwg, 1, dwgalpha, 1);
// 	/* dwgalpha[3, :] = dwg[3, :] */
// 	BLAS_CALL(Dcopy, num_node, dwg + num_node * 3, 1, dwgalpha + num_node * 3, 1);
// 
// 	/* wgalpha = wgold + fact2[0] * dwgold + fact2[1] * dwg */
// 	BLAS_CALL(Dcopy, num_node * BS, wgold, 1, wgalpha, 1);
// 	BLAS_CALL(Daxpy, num_node * BS, fact2 + 0, dwgold, 1, wgalpha, 1);
// 	BLAS_CALL(Daxpy, num_node * BS, fact2 + 1, dwg, 1, wgalpha, 1);
// 	cudaMemset(wgalpha + num_node * 3, 0, num_node * SIZE_OF(f64));
// 
// 
// 	/* Construct the right-hand side */
// 	
// 	start = clock();
// 	AssembleSystem(mesh, wgalpha, dwgalpha, F, NULL, bcs, nbc);
// 	end = clock();
// 	fprintf(stdout, "Assemble F time: %f ms\n", (f64)(end - start) / CLOCKS_PER_SEC * 1000.0);
// 	BLAS_CALL(Dnrm2, num_node * 3, F + num_node * 0, 1, rnorm_init + 0);
// 	BLAS_CALL(Dnrm2, num_node * 1, F + num_node * 3, 1, rnorm_init + 1);
// 	BLAS_CALL(Dnrm2, num_node * 1, F + num_node * 4, 1, rnorm_init + 2);
// 	BLAS_CALL(Dnrm2, num_node * 1, F + num_node * 5, 1, rnorm_init + 3);
// 
// 
// 	CUGUARD(cudaGetLastError());
// 
// 	fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", 0, rnorm_init[0], 1.0, tol);
// 	fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", 0, rnorm_init[1], 1.0, tol);
// 	fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", 0, rnorm_init[2], 1.0, tol);
// 	fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", 0, rnorm_init[3], 1.0, tol);
// 	rnorm_init[0] += 1e-16;
// 	rnorm_init[1] += 1e-16;
// 	rnorm_init[2] += 1e-16;
// 	rnorm_init[3] += 1e-16;
// 
// 	while(!converged && iter < maxit) {
// 		/* Construct the Jacobian matrix */
// 		start = clock();
// 		AssembleSystem(mesh, wgalpha, dwgalpha, NULL, J, bcs, nbc);
// 		end = clock();
// 		fprintf(stdout, "Assemble J time: %f ms\n", (f64)(end - start) / CLOCKS_PER_SEC * 1000.0);
// 		CUGUARD(cudaGetLastError());
// 
// 	
// 		/* Solve the linear system */
// 		cudaMemset(dx, 0, num_node * SIZE_OF(f64) * BS);
// 		// for(index_type ibc = 0; ibc < nbc; ++ibc) {
// 		// 	DirichletApplyVec(bcs[ibc], dx);
// 		// }
// 		cudaStreamSynchronize(0);
// 		start = clock();
// 		KrylovSolve(ksp, J, dx, F);
// 
// 		cudaStreamSynchronize(0);
// 		end = clock();
// 		fprintf(stdout, "Krylov solve time: %f ms\n", (f64)(end - start) / CLOCKS_PER_SEC * 1000.0);
// 
// 		/* Update the solution */	
// 		/* Update dwg by dwg -= dx */
// 		BLAS_CALL(Daxpy, num_node * BS, &minus_one, dx, 1, dwg, 1);
// 		if(0) {
// 			/* Update dwgalpha by dwgalpha[[0,1,2], :] -= kALPHAM * dx[[0,1,2], :] */ 
// 			BLAS_CALL(Daxpy, num_node * 3, fact1 + 3, dx + num_node * 0, 1, dwgalpha + num_node * 0, 1);
// 			/* Update dwgalpha by dwgalpha[3, :] -= dx[3, :] */ 
// 			BLAS_CALL(Daxpy, num_node * 1, &minus_one, dx + num_node * 3, 1, dwgalpha + num_node * 3, 1);
// 			/* Update dwgalpha by dwgalpha[[4,5], :] -= kALPHAM * dx[[4,5], :] */ 
// 			BLAS_CALL(Daxpy, num_node * 2, fact1 + 3, dx + num_node * 4, 1, dwgalpha + num_node * 4, 1);
// 			/* Update wgalpha by wgalpha[[0,1,2], :] -= kDT * kALPHAF * kGAMMA * dx[[0,1,2], :] */
// 			BLAS_CALL(Daxpy, num_node * 3, fact2 + 3, dx + num_node * 0, 1, wgalpha + num_node * 0, 1);
// 			/* Update wgalpha by wgalpha[[4,5], :] -= kDT * kALPHAF * kGAMMA * dx[[4,5], :] */
// 			BLAS_CALL(Daxpy, num_node * 2, fact2 + 3, dx + num_node * 4, 1, wgalpha + num_node * 4, 1);
// 		}
// 		else {
// 			/* dwgalpha = fact1[0] * dwgold + fact1[1] * dwg */ 
// 			cudaMemset(dwgalpha, 0, num_node * SIZE_OF(f64) * BS);
// 			BLAS_CALL(Daxpy, num_node * BS, fact1 + 0, dwgold, 1, dwgalpha, 1);
// 			BLAS_CALL(Daxpy, num_node * BS, fact1 + 1, dwg, 1, dwgalpha, 1);
// 			/* dwgalpha[3, :] = dwg[3, :] */
// 			BLAS_CALL(Dcopy, num_node * 1, dwg + num_node * 3, 1, dwgalpha + num_node * 3, 1);
// 
// 			/* wgalpha = wgold + fact2[0] * dwgold + fact2[1] * dwg */
// 			BLAS_CALL(Dcopy, num_node * BS, wgold, 1, wgalpha, 1);
// 			BLAS_CALL(Daxpy, num_node * BS, fact2 + 0, dwgold, 1, wgalpha, 1);
// 			BLAS_CALL(Daxpy, num_node * BS, fact2 + 1, dwg, 1, wgalpha, 1);
// 			cudaMemset(wgalpha + num_node * 3, 0, num_node * SIZE_OF(f64));
// 
// 		}
// 
// 
// 		start = clock();
// 		AssembleSystem(mesh, wgalpha, dwgalpha, F, NULL, bcs, nbc);
// 		end = clock();
// 		fprintf(stdout, "Assemble F time: %f ms\n", (f64)(end - start) / CLOCKS_PER_SEC * 1000.0);
// 		BLAS_CALL(Dnrm2, num_node * 3, F + num_node * 0, 1, rnorm + 0);
// 		BLAS_CALL(Dnrm2, num_node * 1, F + num_node * 3, 1, rnorm + 1);
// 		BLAS_CALL(Dnrm2, num_node * 1, F + num_node * 4, 1, rnorm + 2);
// 		BLAS_CALL(Dnrm2, num_node * 1, F + num_node * 5, 1, rnorm + 3);
// 		fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", iter + 1, rnorm[0], rnorm[0] / rnorm_init[0], tol);
// 		fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", iter + 1, rnorm[1], rnorm[1] / rnorm_init[1], tol);
// 		fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", iter + 1, rnorm[2], rnorm[2] / rnorm_init[2], tol);
// 		fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", iter + 1, rnorm[3], rnorm[3] / rnorm_init[3], tol);
// 
// 		if (rnorm[0] < tol * rnorm_init[0] &&
// 				rnorm[1] < tol * rnorm_init[1] &&
// 				rnorm[2] < tol * rnorm_init[2] &&
// 				rnorm[3] < tol * rnorm_init[3]) {
// 			converged = TRUE;
// 		}
// 		iter++;
// 
// 	}
// 
// 	CdamFreeDevice(dwgalpha, num_node * SIZE_OF(f64) * BS);
// 	CdamFreeDevice(wgalpha, num_node * SIZE_OF(f64) * BS);
// }

