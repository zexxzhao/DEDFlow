#ifndef __KRYLOV_H__
#define __KRYLOV_H__

// #include <cusparse.h>
#include <cublas_v2.h>
#include "common.h"
#include "matrix.h"

__BEGIN_DECLS__

// typedef void (*PCApplyFunc)(Matrix*, value_type*, value_type*, void*);
typedef void (*KSPSolveFunc)(Matrix*, value_type*, value_type*, void*);
typedef struct Krylov Krylov;
struct Krylov {
	index_type max_iter;
	f64 atol, rtol;
	void* handle;
	// PCApplyFunc pc_apply;
	KSPSolveFunc ksp_solve;
	size_t ksp_ctx_size;
	void* ksp_ctx;

	void* pc;
};

Krylov* KrylovCreateCG(index_type, f64, f64, void*);
Krylov* KrylovCreateGMRES(index_type, f64, f64, void*);
void KrylovDestroy(Krylov* krylov);

void KrylovSolve(Krylov* krylov, Matrix* A, f64* b, f64* x);

typedef struct CdamKrylovOp CdamKrylovOp;
struct CdamKrylovOp {
	void (*alloc) (void**, ptrdiff_t, MemType);
	void (*free) (void*, ptrdiff_t, MemType);
	void (*matvec)(value_type alpha, void* A, void* x, value_type, void* y);
	void (*solve)(void* ksp, void* A, void* x, void* b);
	void (*setup)(void* ksp, void* A, void* config);
	void (*vec_copy)(index_type n, void* x, void* y);
	void (*vec_axpy)(index_type n, value_type alpha, void* x, void* y);
	void (*vec_scal)(index_type n, value_type alpha, void* x);
	void (*vec_dot)(index_type n, void* x, void* y, value_type* result);
	void (*vec_set)(index_type n, void* x, void* y);
	void (*vec_get)(index_type n, void* x, void* y);
	void (*pc_apply)(void* pc, void* x, void* y);
};

typedef struct CdamKrylov CdamKrylov;
struct CdamKrylov {
	index_type min_it, max_it;
	index_type iter;
	value_type rtol, atol;
	value_type rnorm[2];

	void* A;
	void* r;
	ptrdiff_t ctx_size;
	void* ctx;
	void* pc;
	
	CdamKrylovOp op[1];
};
#define CdamKrylovMinIt(krylov) ((krylov)->min_it)
#define CdamKrylovMaxIt(krylov) ((krylov)->max_it)
#define CdamKrylovIter(krylov) ((krylov)->iter)
#define CdamKrylovRtol(krylov) ((krylov)->rtol)
#define CdamKrylovAtol(krylov) ((krylov)->atol)
#define CdamKrylovMatrix(krylov) ((krylov)->A)
#define CdamKrylovResidual(krylov) ((krylov)->r)
#define CdamKrylovPC(krylov) ((krylov)->pc)

void CdamKrylovCreate(CdamKrylov** krylov);
void CdamKrylovDestroy(CdamKrylov* krylov);
void CdamKrylovSetup(CdamKrylov* krylov, void* A, void* config);
void CdamKrylovSolve(CdamKrylov* krylov, void* A, void* x, void* b);


__END_DECLS__
#endif /* __KRYLOV_H__ */
