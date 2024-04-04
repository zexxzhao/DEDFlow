#ifndef __KRYLOV_H__
#define __KRYLOV_H__

#include <cusparse.h>
#include "common.h"

__BEGIN_DECLS__

typedef struct CSRMatrix CSRMatrix;
typedef void (*PCApplyFunc)(CSRMatrix*, f64*, f64*, void*);
typedef void (*KSPSolveFunc)(CSRMatrix*, f64*, f64*, void*);

typedef struct Krylov Krylov;
struct Krylov {
	u32 max_iter;
	f64 atol, rtol;
	cusparseHandle_t handle;
	PCApplyFunc pc_apply;
	KSPSolveFunc ksp_solve;
	size_t ksp_ctx_size, pc_ctx_size;
	void* ksp_ctx;
	void* pc_ctx;
};

Krylov* KrylovCreateCG(u32, f64, f64);
Krylov* KrylovCreateGMRES(u32, f64, f64);
void KrylovDestroy(Krylov* krylov);

void KrylovSolve(Krylov* krylov, CSRMatrix* A, f64* b, f64* x);

__END_DECLS__
#endif /* __KRYLOV_H__ */
