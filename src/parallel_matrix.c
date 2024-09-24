#include "alloc.h"
#include "blas.h"
#include "sequential_matrix.h"
#include "parallel_matrix.h"

__BEGIN_DECLS__

void CdamParMatCreate(MPI_Comm comm, void** A) {
	*A = CdamTMalloc(CdamParMat, 1, HOST_MEM);
	CdamMemset(*A, 0, sizeof(CdamParMat), HOST_MEM);
	((CdamParMat*)*A)->comm = comm;
}
void CdamParMatDestroy(void* A) {
	CdamParMat* mat = (CdamParMat*)A;
	SeqMatDestroy(mat->diag);
	SeqMatDestroy(mat->offd);
	CdamFree(A, sizeof(CdamParMat), HOST_MEM);
}
void CdamParMatSetup(void* A, void* diag, void* offd) {
	index_type nrow = 0, ncol = 0;
	nrow += CdamParMatNumRowExclusive(A);
	nrow += CdamParMatNumRowShared(A);
	nrow += CdamParMatNumRowGhosted(A);
	ncol += CdamParMatNumColExclusive(A);
	ncol += CdamParMatNumColShared(A);
	ncol += CdamParMatNumColGhosted(A);
	CdamParMatDiag(A) = diag;
	CdamParMatOffd(A) = offd;
}


void CdamParMatZero(void* A) {
	SeqMatZero(CdamParMatDiag(A));
	SeqMatZero(CdamParMatOffd(A));
}
void CdamParMatZeroRow(void* A, index_type row, index_type* rows, index_type shift, value_type diag) {
	SeqMatZeroRow(CdamParMatDiag(A), row, rows, shift, diag);
}
void CdamParMatGetSubmat(void* A, index_type nr, index_type* rows, index_type nc, index_type* cols, void* B) {
	index_type nr_exclusive, nr_shared, nr_ghosted;
}

void CdamParMatCopy(void* A, void* B);
void CdamParMatTranspose(void* A, void* B);
void CdamParMatMultAdd(value_type alpha, void* A, value_type* x, value_type beta, value_type* y);
void CdamParMatMatMultAdd(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);
void CdamParMatGetDiag(void* A, value_type* diag, index_type bs);
void CdamParMatAddElemValueBatched(void* A, index_type batch_size, index_type* ien, index_type nshl,
																 index_type block_row, index_type block_col,
																 value_type* value, index_type ldv, index_type stride);


__END_DECLS__
