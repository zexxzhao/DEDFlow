#ifndef __KRYLOV_H__
#define __KRYLOV_H__

// #include <cusparse.h>
#include "common.h"
#include "matrix.h"

__BEGIN_DECLS__

typedef void (*PCApplyFunc)(Matrix*, value_type*, value_type*, void*);
typedef void (*KSPSolveFunc)(Matrix*, value_type*, value_type*, void*);

typedef struct Krylov Krylov;
struct Krylov {
	index_type max_iter;
	f64 atol, rtol;
	PCApplyFunc pc_apply;
	KSPSolveFunc ksp_solve;
	size_t ksp_ctx_size, pc_ctx_size;
	void* ksp_ctx;
	void* pc_ctx;
};

Krylov* KrylovCreateCG(index_type, f64, f64);
Krylov* KrylovCreateGMRES(index_type, f64, f64);
void KrylovDestroy(Krylov* krylov);

void KrylovSolve(Krylov* krylov, Matrix* A, f64* b, f64* x);

__END_DECLS__
#endif /* __KRYLOV_H__ */
