#ifndef __NEWTONSOLVER_H__
#define __NEWTONSOLVER_H__

#include "common.h"

__BEGIN_DECLS__


typedef struct CdamKrylov CdamKrylov;

typedef struct NewtonSolverOps NewtonSolverOps;
typedef void (*NewtonSolverFunctionLinear)(void*, void*, void*);
typedef void (*NewtonSolverFunctionBilinear)(void*, void*, void*);

struct NewtonSolverOps {
	void (*update_solution)(void *data, void *x);
	b32 (*is_converged)(void *data, void *x, b32);
	NewtonSolverFunctionLinear linear_form;
	NewtonSolverFunctionBilinear bilinear_form;
};

typedef struct CdamNewtonSolver CdamNewtonSolver;
struct CdamNewtonSolver {
	MPI_Comm comm;
	value_type rtol;
	value_type atol;
	index_type iter, maxit;
	value_type relaxation;


	CdamKrylov *lsolver;
	void *layout;
	void *A;
	void *dx;
	void *b;

	value_type rnorm_init[16];
	value_type rnorm[16];
	
	NewtonSolverOps op[1];
};

void CdamNewtonSolverCreate(MPI_Comm comm, CdamNewtonSolver **solver);
void CdamNewtonSolverDestroy(CdamNewtonSolver *solver);
void CdamNewtonSolverConfig(CdamNewtonSolver *solver, void* config);

void CdamNewtonSolverSetRelativeTolerance(CdamNewtonSolver *solver, value_type rtol);
void CdamNewtonSolverSetAbsoluteTolerance(CdamNewtonSolver *solver, value_type atol);
void CdamNewtonSolverSetMaxIterations(CdamNewtonSolver *solver, index_type maxit);
void CdamNewtonSolverSetLinearForm(CdamNewtonSolver *solver, NewtonSolverFunctionLinear linear);
void CdamNewtonSolverSetBilinearForm(CdamNewtonSolver *solver, NewtonSolverFunctionBilinear bilinear);
void CdamNewtonSolverSetKrylov(CdamNewtonSolver *solver, CdamKrylov *lsolver);
void CdamNewtonSolverSetLinearSystem(CdamNewtonSolver *solver, void *A, void *x, void *b);

void CdamNewtonSolverGetRelativeTolerance(CdamNewtonSolver *solver, value_type *rtol);
void CdamNewtonSolverGetAbsoluteTolerance(CdamNewtonSolver *solver, value_type *atol);
void CdamNewtonSolverGetMaxIterations(CdamNewtonSolver *solver, index_type *maxit);
void CdamNewtonSolverGetLinearSolver(CdamNewtonSolver *solver, CdamKrylov **lsolver);
// void CdamNewtonSolverGetLinearSystem(CdamNewtonSolver *solver, void **A, void **x, void **b);

void CdamNewtonSolverSolve(CdamNewtonSolver *solver, value_type *x);

#define CdamNewtonSolverA(solver) ((solver)->A)
#define CdamNewtonSolverX(solver) ((solver)->dx)
#define CdamNewtonSolverB(solver) ((solver)->b)

__END_DECLS__


#endif /* __NEWTONSOLVER_H__ */
