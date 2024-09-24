#include "alloc.h"
#include "blas.h"
#include "commutor.h"
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
void CdamParMatSetup(void* A) {
}


void CdamParMatZero(void* A) {
	SeqMatZero(CdamParMatDiag(A));
	if(CdamParMatAssemblyType(A) == MAT_ASSEMBLED) {
		SeqMatZero(CdamParMatOffd(A));
	}
}
void CdamParMatZeroRow(void* A, index_type row, index_type* rows, index_type shift, value_type diag) {
	index_type i, n_owned = 0;
	index_type nrow_owned = CdamParMatNumRowOwned(A);
	if(CdamParMatAssemblyType(A) == MAT_ASSEMBLED) {
		SeqMatZeroRow(CdamParMatDiag(A), row, rows, shift, diag);
		SeqMatZeroRow(CdamParMatOffd(A), row, rows, shift, 0.0);
	}
	else {
		/* Count rows[:] < n_owned as n_owned */
		index_type* h_rows = CdamTMalloc(index_type, nrow_owned, HOST_MEM);
		CdamMemcpy(h_rows, rows, nrow_owned*sizeof(index_type), HOST_MEM, DEVICE_MEM);
		for(i = 0; i < row; ++i) {
			if(h_rows[i] < n_owned) {
				n_owned++;
			}
		}
		
		SeqMatZeroRow(CdamParMatDiag(A), n_owned, rows, shift, diag);
		SeqMatZeroRow(CdamParMatDiag(A), row - n_owned, rows + n_owned, shift, 0.0);
		CdamFree(h_rows, nrow_owned*sizeof(index_type), HOST_MEM);
	}
}
void CdamParMatGetSubmat(void* A, index_type nr, index_type* rows, index_type nc, index_type* cols, void* B, void** auxiliary) {
	index_type nr_exclusive = 0, nr_shared = 0, nr_ghosted = 0;
	index_type nc_exclusive = 0, nc_shared = 0, nc_ghosted = 0;
	index_type i;

	index_type *h_rows, *h_cols;
	h_rows = CdamTMalloc(index_type, nr, HOST_MEM);
	h_cols = CdamTMalloc(index_type, nc, HOST_MEM);
	CdamMemcpy(h_rows, rows, nr*sizeof(index_type), HOST_MEM, DEVICE_MEM);
	CdamMemcpy(h_cols, cols, nc*sizeof(index_type), HOST_MEM, DEVICE_MEM);

	for(i = 0; i < nr; ++i) {
		if(h_rows[i] < CdamParMatNumRowExclusive(A)) {
			nr_exclusive++;
		} else if(h_rows[i] < CdamParMatNumRowExclusive(A) + CdamParMatNumRowShared(A)) {
			nr_shared++;
		} else {
			nr_ghosted++;
		}
	}
	for(i = 0; i < nc; ++i) {
		if(h_cols[i] < CdamParMatNumColExclusive(A)) {
			nc_exclusive++;
		} else if(h_cols[i] < CdamParMatNumColExclusive(A) + CdamParMatNumColShared(A)) {
			nc_shared++;
		} else {
			nc_ghosted++;
		}
	}

	CdamParMat* mata = (CdamParMat*)A;
	CdamParMat* matb = (CdamParMat*)B;
	matb->comm = mata->comm;
	CdamParMatNumRowExclusive(B) = nr_exclusive;
	CdamParMatNumRowShared(B) = nr_shared;
	CdamParMatNumRowGhosted(B) = nr_ghosted;

	CdamParMatNumColExclusive(B) = nc_exclusive;
	CdamParMatNumColShared(B) = nc_shared;
	CdamParMatNumColGhosted(B) = nc_ghosted;

	SeqMatGetSubmat(CdamParMatDiag(A), nr, rows, nc, cols, CdamParMatDiag(B), auxiliary);


	CdamFree(h_rows, nr*sizeof(index_type), HOST_MEM);
	CdamFree(h_cols, nc*sizeof(index_type), HOST_MEM);
}

void CdamParMatCopy(void* A, void* B) {
	CdamParMat* mata = (CdamParMat*)A;
	CdamParMat* matb = (CdamParMat*)B;
	matb->comm = mata->comm;
	CdamParMatNumRowExclusive(B) = CdamParMatNumRowExclusive(A);
	CdamParMatNumRowShared(B) = CdamParMatNumRowShared(A);
	CdamParMatNumRowGhosted(B) = CdamParMatNumRowGhosted(A);

	CdamParMatNumColExclusive(B) = CdamParMatNumColExclusive(A);
	CdamParMatNumColShared(B) = CdamParMatNumColShared(A);
	CdamParMatNumColGhosted(B) = CdamParMatNumColGhosted(A);

	SeqMatCopy(CdamParMatDiag(A), CdamParMatDiag(B));
	SeqMatCopy(CdamParMatOffd(A), CdamParMatOffd(B));
}
void CdamParMatTranspose(void* A, void* B) {
	if(B && A != B) {
		CdamParMat* mata = (CdamParMat*)A;
		CdamParMat* matb = (CdamParMat*)B;
		matb->comm = mata->comm;
		CdamParMatNumRowExclusive(B) = CdamParMatNumColExclusive(A);
		CdamParMatNumRowShared(B) = CdamParMatNumColShared(A);
		CdamParMatNumRowGhosted(B) = CdamParMatNumColGhosted(A);

		CdamParMatNumColExclusive(B) = CdamParMatNumRowExclusive(A);
		CdamParMatNumColShared(B) = CdamParMatNumRowShared(A);
		CdamParMatNumColGhosted(B) = CdamParMatNumRowGhosted(A);

		SeqMatTranspose(CdamParMatDiag(A), CdamParMatDiag(B));
		SeqMatTranspose(CdamParMatOffd(A), CdamParMatOffd(B));
	} else {
		SeqMatTranspose(CdamParMatDiag(A), CdamParMatDiag(A));
		SeqMatTranspose(CdamParMatOffd(A), CdamParMatOffd(A));
	}
}
void CdamParMatMultAdd(value_type alpha, void* A, value_type* x, value_type beta, value_type* y) {
	index_type i;
	SeqMatMultAdd(alpha, CdamParMatDiag(A), x, beta, y);
	SeqMatMultAdd(alpha, CdamParMatOffd(A), x, 1.0, y);
	
	if(CdamParMatAssemblyType(A) == MAT_DISASSEMBLED) {
		CdamCommutor* commu = (CdamCommutor*)CdamParMatCommutor(A);
		int n = commu->num_node_local;
		/* Assume the DOFs is colunm-wise */
		for(i = 0; i < 6; i++) {
			CdamCommutorForward(commu, y + i * n, sizeof(value_type));
			CdamCommutorBackward(commu, y + i * n, sizeof(value_type));
		}
	}

}
void CdamParMatMatMultAdd(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	SeqMatMatMultAdd(alpha, CdamParMatDiag(A), CdamParMatDiag(B), beta, CdamParMatDiag(C), reuse);
	SeqMatMatMultAdd(alpha, CdamParMatOffd(A), CdamParMatOffd(B), 1.0, CdamParMatDiag(C), MAT_REUSE);
}
void CdamParMatGetDiag(void* A, value_type* diag, index_type bs) {
	SeqMatGetDiag(CdamParMatDiag(A), diag, bs);
	if(CdamParMatAssemblyType(A) == MAT_DISASSEMBLED) {
		CdamCommutor* commu = (CdamCommutor*)CdamParMatCommutor(A);
		int n = commu->num_node_local;
		/* Assume the DOFs is colunm-wise */
		CdamCommutorForward(commu, diag, bs * bs * (int)sizeof(value_type));
		CdamCommutorBackward(commu, diag, bs * bs * (int)sizeof(value_type));
	}
}
void CdamParMatAddElemValueBatched(void* A, index_type batch_size, index_type* ien, index_type nshl,
																 index_type block_row, index_type block_col,
																 value_type* value, index_type ldv, index_type stride) {
	SeqMatAddElemValueBatched(CdamParMatDiag(A), batch_size, ien, nshl, block_row, block_col, value, ldv, stride);
}


__END_DECLS__
