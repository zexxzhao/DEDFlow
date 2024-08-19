#ifndef __NEWTONSOLVER_H__
#define __NEWTONSOLVER_H__

#include "common.h"

__BEGIN_DECLS__


typedef struct CDAM_LinearSolver CDAM_LinearSolver;
typedef struct CDAM_Form CDAM_Form;
typedef struct CDAM_Vec CDAM_Vec;
typedef struct CDAM_Mat CDAM_Mat;

typedef struct NewtonSolverOps NewtonSolverOps;
struct NewtonSolverOps {
	void (*update_solution)(void *data, CDAM_Vec *x);
	b32 (*converged)(void *data, CDAM_Vec *x);
};

typedef struct CDAM_NewtonSolver CDAM_NewtonSolver;
struct CDAM_NewtonSolver {
	MPI_Comm comm;
	value_type rtol;
	value_type atol;
	index_type iter, maxit;
	value_type relaxation;

	CDAM_Form *fem;
	CDAM_LinearSolver *lsolver;
	CDAM_Mat *A;
	CDAM_Vec *x;
	CDAM_Vec *b;

	value_type *rnorm_init;
	value_type *rnorm;
	
	NewtonSolverOps op[1];
};

void CDAM_NewtonSolverCreate(MPI_Comm comm, CDAM_NewtonSolver **solver);
void CDAM_NewtonSolverDestroy(CDAM_NewtonSolver *solver);
void CDAM_NewtonSolverConfig(CDAM_NewtonSolver *solver, void* config);

void CDAM_NewtonSolverSetRelativeTolerance(CDAM_NewtonSolver *solver, value_type rtol);
void CDAM_NewtonSolverSetAbsoluteTolerance(CDAM_NewtonSolver *solver, value_type atol);
void CDAM_NewtonSolverSetMaxIterations(CDAM_NewtonSolver *solver, index_type maxit);
void CDAM_NewtonSolverSetFEM(CDAM_NewtonSolver *solver, CDAM_Form *fem);
void CDAM_NewtonSolverSetLinearSolver(CDAM_NewtonSolver *solver, CDAM_LinearSolver *lsolver);
void CDAM_NewtonSolverSetLinearSystem(CDAM_NewtonSolver *solver, CDAM_Mat *A, CDAM_Vec *x, CDAM_Vec *b);

void CDAM_NewtonSolverGetRelativeTolerance(CDAM_NewtonSolver *solver, value_type *rtol);
void CDAM_NewtonSolverGetAbsoluteTolerance(CDAM_NewtonSolver *solver, value_type *atol);
void CDAM_NewtonSolverGetMaxIterations(CDAM_NewtonSolver *solver, index_type *maxit);
void CDAM_NewtonSolverGetFEM(CDAM_NewtonSolver *solver, CDAM_Form **fem);
void CDAM_NewtonSolverGetLinearSolver(CDAM_NewtonSolver *solver, CDAM_LinearSolver **lsolver);
void CDAM_NewtonSolverGetLinearSystem(CDAM_NewtonSolver *solver, CDAM_Mat **A, CDAM_Vec **x, CDAM_Vec **b);

void CDAM_NewtonSolverSolve(CDAM_NewtonSolver *solver, CDAM_Vec *x);

__END_DECLS__


#endif /* __NEWTONSOLVER_H__ */
