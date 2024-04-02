#ifndef __KRYLOV_H__
#define __KRYLOV_H__

#include <cusparse.h>
#include "common.h"

__BEGIN_DECLS__
typedef struct CSRMatrix CSRMatrix;
typedef struct Krylov Krylov;
struct Krylove {
	u32 max_iter;
	f64 atol, rtol;
	cusparseHandle_t handle;
	void* buffer_mv;
	void(*solve)(CSRMatrix* A, f64* b, f64* x);
	void(*pc_solve)(CSRMatrix* A, f64* b, f64* x);
};

Krylov* KrylovCreateCG(u32, f64, f64);
Krylov* KrylovCreateGMRES(u32, f64, f64);
void KrylovDestroy(Krylov* krylov);

__END_DECLS__
#endif /* __KRYLOV_H__ */
