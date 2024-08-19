#include <mpi.h>
#include "json.h"
#include "vec.h"
#include "NewtonSolver.h"

static b32 CheckConvergencePrivate(void* data, CDAM_Vec* x, b32 verbose) {
	CDAM_NewtonSolver* solver = (CDAM_NewtonSolver*)data;

	value_type atol = solver->atol;
	value_type rtol = solver->rtol;
	CDAM_Form* fem = solver->fem;
	
	value_type *b = CDAM_VecRawPtr(x, 0);	

	index_type i;
	index_type n = CDAM_VecNumComponents(x); 

	CDAM_Vec* v;
	value_type* r = solver->rnorm;
	value_type* r0 = solver->rnorm_init, r0norm = 0.0;

	for(i = 0; i < n; i++) {
		v = &WrapVec(CDAM_VecSub(x)[i]);
		CDAM_VecNorm(v, r + i);
	}

	for(i = 0; i < n; i++) {
		r0norm += r0[i] * r0[i];
	}
	if(r0norm == 0.0) {
		memcpy(r0, r, n * sizeof(value_type));
	}

	b32 converged = 1;
	for(i = 0; i < n; i++) {
		converged = converged && (r[i] < atol || r[i] < rtol * r0[i]);
	}

	if(verbose) {
		for(i = 0; i < n; ++i) {
			fprintf(stdout, "Newton %d) abs=%.17e rel=%8.6e (tol=%8.6e)\n", solver->iter, r[i], r[i] / r0[i], rtol);
		}
	}
	return converged;
}

static void UpdateSolutionPrivate(void* data, CDAM_Vec* dx) {
	NewtonSolver* solver = (NewtonSolver*)data;
	CDAM_Form* fem = solver->fem;
	value_type* x;
	CDAM_Mesh* mesh;
	value_type minus_one = -1.0;	
	index_type num_node, bs;
	
	CDAM_FormGetSolution(fem, &x);

	CDAM_FormGetMesh(fem, &mesh);
	num_node = CDAM_MeshNumNodes(mesh);
	bs = CDAM_FormNumComponents(fem);


	cublasHandle_t handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE);

	cublasDaxpy(handle, num_node * bs, &minus_one, CDAM_VecRawPtr(dx, 0), 1, x, 1);
	
}

void CDAM_NewtonSolverCreate(MPI_Comm comm, CDAM_NewtonSolver **solver) {
	*solver = (CDAM_NewtonSolver*)CdamMallocHost(sizeof(CDAM_NewtonSolver));
	memset(*solver, 0, sizeof(CDAM_NewtonSolver));
	(*solver)->comm = comm;
	(*solver)->max_iter = 100;
	(*solver)->rtol = 1.0e-6;
	(*solver)->atol = 1.0e-12;
	(*solver)->relaxation = 1.0;

	(*solver)->op->CheckConvergence = CheckConvergencePrivate;
	(*solver)->op->UpdateSolution = UpdateSolutionPrivate;
}

void CDAM_NewtonSolverDestroy(CDAM_NewtonSolver *solver) {
	CdamFreeHost(solver->rnorm_init, sizeof(value_type) * 2 * CDAM_FormNumComponents(solver->fem));
	CdamFreeHost(solver);
	*solver = NULL;
}

void CDAM_NewtonSolverConfig(CDAM_NewtonSolver *solver, void* config) {
	cJSON *json = (cJSON*)config;
	solver->rtol = JSONGetItem(json, "RelativeTolerance")->valuedouble;
	solver->atol = JSONGetItem(json, "AbsoluteTolerance")->valuedouble;
	solver->max_iter = JSONGetItem(json, "MaxIterations")->valueint;
	solver->relaxation = JSONGetItem(json, "RelaxationFactor")->valuedouble;

}


void CDAM_NewtonSolverSetRelativeTolerance(CDAM_NewtonSolver *solver, value_type rtol) {
	solver->rtol = rtol;
}

void CDAM_NewtonSolverSetAbsoluteTolerance(CDAM_NewtonSolver *solver, value_type atol) {
	solver->atol = atol;
}

void CDAM_NewtonSolverSetMaxIterations(CDAM_NewtonSolver *solver, int max_iter) {
	solver->max_iter = max_iter;
}

void CDAM_NewtonSolverSetFEM(CDAM_NewtonSolver *solver, CDAM_Form *fem) {
	solver->fem = fem;
	index_type bs = CDAM_FormNumComponents(fem);
	index_type num_node;;
	CDAM_Mesh* mesh;
	CDAM_FormGetMesh(fem, &mesh);
	solver->rnorm_init = (value_type*)CdamMallocHost(2 * n * sizeof(value_type));
	solver->rnorm = solver->rnorm_init + n;
	memset(solver->rnorm_init, 0, 2 * n * sizeof(value_type));

	num_node = CDAM_MeshNumNodes(mesh);
}

void CDAM_NewtonSolverSetLinearSolver(CDAM_NewtonSolver *solver, CDAM_LinearSolver *lsolver) {
	solver->lsolver = lsolver;
}

void CDAM_NewtonSolverSetLinearSystem(CDAM_NewtonSolver *solver, CDAM_Mat* A, CDAM_Vec* x, CDAM_Vec* b) {
	solver->A = A;
	solver->x = x;
	solver->b = b;
}


void CDAM_NewtonSolverGetRelativeTolerance(CDAM_NewtonSolver *solver, value_type *rtol) {
	if(rtol)
		*rtol = solver->rtol;
}
void CDAM_NewtonSolverGetAbsoluteTolerance(CDAM_NewtonSolver *solver, value_type *atol) {
	if(atol)
		*atol = solver->atol;
}
void CDAM_NewtonSolverGetMaxIterations(CDAM_NewtonSolver *solver, int *max_iter) {
	if(max_iter)
		*max_iter = solver->max_iter;
}
void CDAM_NewtonSolverGetFEM(CDAM_NewtonSolver *solver, CDAM_Form **fem) {
	if(fem)
		*fem = solver->fem;
}
void CDAM_NewtonSolverGetLinearSolver(CDAM_NewtonSolver *solver, CDAM_LinearSolver **lsolver) {
	if(lsolver)
		*lsolver = solver->lsolver;
}
void CDAM_NewtonSolverGetLinearSystem(CDAM_NewtonSolver *solver, CDAM_Mat** A, CDAM_Vec** x, CDAM_Vec** b) {
	if(A) *A = solver->A;
	if(x) *x = solver->x;
	if(b) *b = solver->b;
}



void CDAM_NewtonSolverSolve(CDAM_NewtonSolver* solver) {
	CDAM_Form* fem = solver->fem;
	CDAM_LinearSolver* lsolver = solver->lsolver;
	CDAM_Mat* A = solver->A;
	CDAM_Vec* b = solver->b;
	CDAM_Vec* x = solver->x;
	CDAM_Mesh* mesh;
	value_type rtol = solver->rtol;
	value_type atol = solver->atol;
	value_type relaxation = solver->relaxation;
	index_type max_iter = solver->max_iter;

	index_type i;
	index_type num_node, bs;

	b32 converged = 0;

	value_type* dwgold, *wgold, *dwg;

	f64 fact1[] = {1.0 - kALPHAM, kALPHAM, kALPHAM - 1.0, -kALPHAM};
	f64 fact2[] = {kDT * kALPHAF * (1.0 - kGAMMA), kDT * kALPHAF * kGAMMA,
								 -kDT * kALPHAF * (1.0 - kGAMMA), -kDT * kALPHAF * kGAMMA};

	cublasHandle_t handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE);

	
	CDAM_FormGetMesh(fem, &mesh);

	num_node = CDAM_MeshNumNodes(mesh);
	bs = CDAM_FormNumComponents(fem);
	
	/* Construct the right-hand side */
	CDAM_FormEvalResidual(fem, b);


	solver->iter = 0;
	converged = solver->op->CheckConvergence(solver, b, 1);

	while(not converged || solver->iter < max_iter) {
		/* Construct the Jacobian matrix */
		CDAM_FormEvalJacobian(fem, A);

		/* Solve the linear system */
		CDAM_KrylovSolve(lsolver, A, x, b);
		
		/* Update the solution */
		solver->op->UpdateSolution(solver, x);

		/* Construct the right-hand side */
		CDAM_FormEvalResidual(fem, b);

		converged = solver->op->CheckConvergence(solver, b, 1);
		if(converged) {
			break;
		}
		solver->iter++;
	}

}
