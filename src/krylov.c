#include "alloc.h"
#include "krylov.h"

__BEGIN_DECLS__

static Krylov* KryloveInitPrivate(u32 max_iter, f64 atol, f64 rtol) {
	Krylov* ksp = (Krylov*)CdamMallocHost(sizeof(Krylov));
	ksp->handle = NULL;
	ksp->A = NULL;
	ksp->solve = NULL;
}

static void CGSolvePrivate(CSRMatrix* A, f64* x, f64* b) {
	u32 n = CSRMatrixNumRows(A);
	cusparseDnVecDescr_t vec_x, vec_b;
	cusparseCreateDnVec(&vec_x, n, x);
	cusparseCreateDnVec(&vec_b, n, b);

	cusparseDnMatDescr_t mat_A = CSRMatrixDescr(A);
	/* TODO: Implement CG solver */
}

static void GMRESSolvePrivate(CSRMatrix* A, f64* x, f64* b) {
	u32 n = CSRMatrixNumRows(A);
	cusparseDnVecDescr_t vec_x, vec_b;
	cusparseCreateDnVec(&vec_x, n, x);
	cusparseCreateDnVec(&vec_b, n, b);

	cusparseDnMatDescr_t mat_A = CSRMatrixDescr(A);


}

Krylov* KryloveCreateCG(u32 max_iter, f64 atol, f64 rtol) {
	Krylov* ksp = KryloveInitPrivate(max_iter, atol, rtol);
	cusparseCreate(&ksp->handle);
}

__END_DECLS__
