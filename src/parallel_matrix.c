#include "alloc.h"
#include "blas.h"
#include "commutor.h"
#include "indexing.h"
#include "sequential_matrix.h"
#include "parallel_matrix.h"

__BEGIN_DECLS__

void CdamParMatCreate(MPI_Comm comm, void** A) {
	*A = CdamTMalloc(CdamParMat, 1, HOST_MEM);
	CdamMemset(*A, 0, sizeof(CdamParMat), HOST_MEM);
	((CdamParMat*)*A)->comm = comm;
}

void CdamParMatDestroy(void* A) {

	SeqMatDestroy(CdamParMatII(A));
	SeqMatDestroy(CdamParMatIS(A));
	SeqMatDestroy(CdamParMatIG(A));

	SeqMatDestroy(CdamParMatSI(A));
	SeqMatDestroy(CdamParMatSS(A));
	SeqMatDestroy(CdamParMatSG(A));

	SeqMatDestroy(CdamParMatGI(A));
	SeqMatDestroy(CdamParMatGS(A));
	SeqMatDestroy(CdamParMatGG(A));


	CdamFree(A, sizeof(CdamParMat), HOST_MEM);
}

void CdamParMatSetup(void* A) {
}

void CdamParMatZero(void* A) {
	SeqMatZero(CdamParMatII(A));
	SeqMatZero(CdamParMatIS(A));
	SeqMatZero(CdamParMatIG(A));

	SeqMatZero(CdamParMatSI(A));
	SeqMatZero(CdamParMatSS(A));
	SeqMatZero(CdamParMatSG(A));

	SeqMatZero(CdamParMatGI(A));
	SeqMatZero(CdamParMatGS(A));
	SeqMatZero(CdamParMatGG(A));
}

void CdamParMatZeroRow(void* A, index_type row, index_type* rows, value_type diag) {
	index_type i, n_owned = 0;
	index_type nrow_owned = CdamParMatNumRowOwned(A);
	index_type nrow_exclusive = CdamParMatNumRowExclusive(A);
	index_type nrow_shared = CdamParMatNumRowShared(A);
	index_type nrow_ghosted = CdamParMatNumRowGhosted(A);

	SeqMatZeroRow(CdamParMatII(A), row, rows, 0, diag);
	SeqMatZeroRow(CdamParMatIS(A), row, rows, 0, 0.0);
	SeqMatZeroRow(CdamParMatIG(A), row, rows, 0, 0.0);

	SeqMatZeroRow(CdamParMatSI(A), row, rows, -nrow_exlusive, 0.0);
	SeqMatZeroRow(CdamParMatSS(A), row, rows, -nrow_exclusive, diag);
	SeqMatZeroRow(CdamParMatSG(A), row, rows, -nrow_exclusive, 0.0);

	if(CdamParMatAssemblyType(A) == MAT_DISASSEMBLED) {
		SeqMatZeroRow(CdamParMatGI(A), row, rows, -nrow_exclusive - nrow_shared, 0.0);
		SeqMatZeroRow(CdamParMatGS(A), row, rows, -nrow_exclusive - nrow_shared, 0.0);
		SeqMatZeroRow(CdamParMatGG(A), row, rows, -nrow_exclusive - nrow_shared, 0.0);
	}
	else {
		SeqMatZero(CdamParMatGI(A));
		SeqMatZero(CdamParMatGS(A));
		SeqMatZero(CdamParMatGG(A));
	}

}
void CdamParMatGetSubmat(void* A, index_type nr, index_type* rows, index_type nc, index_type* cols, void* B) {
	index_type nr_exclusive = 0, nr_shared = 0, nr_ghosted = 0;
	index_type nrow_offset[4] = {0, 0, 0, 0};
	index_type nc_exclusive = 0, nc_shared = 0, nc_ghosted = 0;
	index_type ncol_offset[4] = {0, 0, 0, 0};
	index_type i;

	nrow_offset[0] = 0;
	nrow_offset[1] = CdamParMatNumRowExclusive(A);
	nrow_offset[2] = nr_offset[1] + CdamParMatNumRowShared(A);
	nrow_offset[3] = nr_offset[2] + CdamParMatNumRowGhosted(A);

	nr_exclusive = CountIndexByRangeGPU(nr, rows, nrow_offset[0], nrow_offset[1]);
	nr_shared = CountIndexByRangeGPU(nr, rows, nrow_offset[1], nrow_offset[2]);
	nr_ghosted = CountIndexByRangeGPU(nr, rows, nrow_offset[2], nrow_offset[3]);

	ncol_offset[0] = 0;
	ncol_offset[1] = CdamParMatNumColExclusive(A);
	ncol_offset[2] = ncol_offset[1] + CdamParMatNumColShared(A);
	ncol_offset[3] = ncol_offset[2] + CdamParMatNumColGhosted(A);

	nc_exclusive = CountIndexByRangeGPU(nc, cols, ncol_offset[0], ncol_offset[1]);
	nc_shared = CountIndexByRangeGPU(nc, cols, ncol_offset[1], ncol_offset[2]);
	nc_ghosted = CountIndexByRangeGPU(nc, cols, ncol_offset[2], ncol_offset[3]);

	CdamParMat* mata = (CdamParMat*)A;
	CdamParMat* matb = (CdamParMat*)B;
	matb->comm = mata->comm;
	CdamParMatNumRowExclusive(B) = nr_exclusive;
	CdamParMatNumRowShared(B) = nr_shared;
	CdamParMatNumRowGhosted(B) = nr_ghosted;

	CdamParMatNumColExclusive(B) = nc_exclusive;
	CdamParMatNumColShared(B) = nc_shared;
	CdamParMatNumColGhosted(B) = nc_ghosted;

	SeqMatGetSubmat(CdamParMatII(A), nr, rows, nc, cols, CdamParMatII(B));
	SeqMatGetSubmat(CdamParMatIS(A), nr, rows, nc, cols, CdamParMatIS(B));
	SeqMatGetSubmat(CdamParMatIG(A), nr, rows, nc, cols, CdamParMatIG(B));

	SeqMatGetSubmat(CdamParMatSI(A), nr, rows, nc, cols, CdamParMatSI(B));
	SeqMatGetSubmat(CdamParMatSS(A), nr, rows, nc, cols, CdamParMatSS(B));
	SeqMatGetSubmat(CdamParMatSG(A), nr, rows, nc, cols, CdamParMatSG(B));

	SeqMatGetSubmat(CdamParMatGI(A), nr, rows, nc, cols, CdamParMatGI(B));
	SeqMatGetSubmat(CdamParMatGS(A), nr, rows, nc, cols, CdamParMatGS(B));
	SeqMatGetSubmat(CdamParMatGG(A), nr, rows, nc, cols, CdamParMatGG(B));

}

// void CdamParMatCopy(void* A, void* B) {
// 	CdamParMat* mata = (CdamParMat*)A;
// 	CdamParMat* matb = (CdamParMat*)B;
// 	matb->comm = mata->comm;
// 	CdamParMatNumRowExclusive(B) = CdamParMatNumRowExclusive(A);
// 	CdamParMatNumRowShared(B) = CdamParMatNumRowShared(A);
// 	CdamParMatNumRowGhosted(B) = CdamParMatNumRowGhosted(A);
// 
// 	CdamParMatNumColExclusive(B) = CdamParMatNumColExclusive(A);
// 	CdamParMatNumColShared(B) = CdamParMatNumColShared(A);
// 	CdamParMatNumColGhosted(B) = CdamParMatNumColGhosted(A);
// 
// 	SeqMatCopy(CdamParMatDiag(A), CdamParMatDiag(B));
// 	SeqMatCopy(CdamParMatOffd(A), CdamParMatOffd(B));
// }

static void SwapVoidPtr(void** a, void** b) {
	void* tmp = *a;
	*a = *b;
	*b = tmp;
}
void CdamParMatTranspose(void* A) {
	SeqMatTranspose(CdamParMatII(A));
	SeqMatTranspose(CdamParMatIS(A));
	SeqMatTranspose(CdamParMatIG(A));

	SeqMatTranspose(CdamParMatSI(A));
	SeqMatTranspose(CdamParMatSS(A));
	SeqMatTranspose(CdamParMatSG(A));

	SeqMatTranspose(CdamParMatGI(A));
	SeqMatTranspose(CdamParMatGS(A));
	SeqMatTranspose(CdamParMatGG(A));

	SwapVoidPtr(&CdamParMatIS(A), &CdamParMatSI(A));
	SwapVoidPtr(&CdamParMatIG(A), &CdamParMatGI(A));
	SwapVoidPtr(&CdamParMatSG(A), &CdamParMatGS(A));
}

void CdamParMatMultAdd(value_type alpha, void* A, value_type* x, value_type beta, value_type* y) {
	index_type nrow_offset[4] = {0, 0, 0, 0};
	index_type ncol_offset[4] = {0, 0, 0, 0};

	nrow_offset[0] = 0;
	nrow_offset[1] = CdamParMatNumRowExclusive(A);
	nrow_offset[2] = nrow_offset[1] + CdamParMatNumRowShared(A);
	nrow_offset[3] = nrow_offset[2] + CdamParMatNumRowGhosted(A);

	ncol_offset[0] = 0;
	ncol_offset[1] = CdamParMatNumColExclusive(A);
	ncol_offset[2] = ncol_offset[1] + CdamParMatNumColShared(A);
	ncol_offset[3] = ncol_offset[2] + CdamParMatNumColGhosted(A);

	SeqMatMultAdd(alpha, CdamParMatII(A), x + ncol_offset[0], beta, y + nrow_offset[0]);
	SeqMatMultAdd(alpha, CdamParMatIS(A), x + ncol_offset[1], 1.0, y + nrow_offset[0]);
	SeqMatMultAdd(alpha, CdamParMatIG(A), x + ncol_offset[2], 1.0, y + nrow_offset[0]);

	SeqMatMultAdd(alpha, CdamParMatSI(A), x + ncol_offset[0], beta, y + nrow_offset[1]);
	SeqMatMultAdd(alpha, CdamParMatSS(A), x + ncol_offset[1], 1.0, y + nrow_offset[1]);
	SeqMatMultAdd(alpha, CdamParMatSG(A), x + ncol_offset[2], 1.0, y + nrow_offset[1]);

	if(CdamParMatAssemblyType(A) == MAT_DISASSEMBLED) {
		SeqMatMultAdd(alpha, CdamParMatGI(A), x + ncol_offset[0], beta, y + nrow_offset[2]);
		SeqMatMultAdd(alpha, CdamParMatGS(A), x + ncol_offset[1], 1.0, y + nrow_offset[2]);
		SeqMatMultAdd(alpha, CdamParMatGG(A), x + ncol_offset[2], 1.0, y + nrow_offset[2]);
	}
	
	if(CdamParMatAssemblyType(A) == MAT_DISASSEMBLED) {
		CdamCommutor* commu = (CdamCommutor*)CdamParMatCommutor(A);
		CdamLayout* map = CdamParMatRowMap(A);
		CdamCommuFordward(commu, y, map, sizeof(value_type));
		CdamCommuBackword(commu, y, map, sizeof(value_type));
	}

}

void CdamParMatMultTransposeAdd(value_type alpha, void* A, value_type* x, value_type beta, value_type* y) {
	index_type nrow_offset[4] = {0, 0, 0, 0};
	index_type ncol_offset[4] = {0, 0, 0, 0};

	nrow_offset[0] = 0;
	nrow_offset[1] = CdamParMatNumRowExclusive(A);
	nrow_offset[2] = nrow_offset[1] + CdamParMatNumRowShared(A);
	nrow_offset[3] = nrow_offset[2] + CdamParMatNumRowGhosted(A);

	ncol_offset[0] = 0;
	ncol_offset[1] = CdamParMatNumColExclusive(A);
	ncol_offset[2] = ncol_offset[1] + CdamParMatNumColShared(A);
	ncol_offset[3] = ncol_offset[2] + CdamParMatNumColGhosted(A);

	SeqMatMultTransposeAdd(alpha, CdamParMatII(A), x + nrow_offset[0], beta, y + ncol_offset[0]);
	SeqMatMultTransposeAdd(alpha, CdamParMatIS(A), x + nrow_offset[0], beta, y + ncol_offset[1]);
	SeqMatMultTransposeAdd(alpha, CdamParMatIG(A), x + nrow_offset[0], beta, y + ncol_offset[2]);

	SeqMatMultTransposeAdd(alpha, CdamParMatSI(A), x + nrow_offset[1], 1.0, y + ncol_offset[0]);
	SeqMatMultTransposeAdd(alpha, CdamParMatSS(A), x + nrow_offset[1], 1.0, y + ncol_offset[1]);
	SeqMatMultTransposeAdd(alpha, CdamParMatSG(A), x + nrow_offset[1], 1.0, y + ncol_offset[2]);

	SeqMatMultTransposeAdd(alpha, CdamParMatGI(A), x + nrow_offset[2], 1.0, y + ncol_offset[0]);
	SeqMatMultTransposeAdd(alpha, CdamParMatGS(A), x + nrow_offset[2], 1.0, y + ncol_offset[1]);
	SeqMatMultTransposeAdd(alpha, CdamParMatGG(A), x + nrow_offset[2], 1.0, y + ncol_offset[2]);

	if(CdamParMatAssemblyType(A) == MAT_DISASSEMBLED) {
		CdamCommutor* commu = (CdamCommutor*)CdamParMatCommutor(A);
		CdamLayout* map = CdamParMatColMap(A);
		CdamCommuFordward(commu, y, map, sizeof(value_type));
		CdamCommuBackword(commu, y, map, sizeof(value_type));
	}
}

void CdamParMatMatMultAdd(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse) {
	SeqMatMatMultAdd(alpha, CdamParMatII(A), CdamParMatII(B), beta, CdamParMatII(C), reuse);
	SeqMatMatMultAdd(alpha, CdamParMatIS(A), CdamParMatSI(B), 1.0, CdamParMatII(C), MAT_REUSE);
	SeqMatMatMultAdd(alpha, CdamParMatIG(A), CdamParMatGI(B), 1.0, CdamParMatII(C), MAT_REUSE);

	SeqMatMatMultAdd(alpha, CdamParMatII(A), CdamParMatIS(B), beta, CdamParMatIS(C), reuse);
	SeqMatMatMultAdd(alpha, CdamParMatIS(A), CdamParMatSS(B), 1.0, CdamParMatIS(C), MAT_REUSE);
	SeqMatMatMultAdd(alpha, CdamParMatIG(A), CdamParMatGS(B), 1.0, CdamParMatIS(C), MAT_REUSE);

	SeqMatMatMultAdd(alpha, CdamParMatII(A), CdamParMatIG(B), beta, CdamParMatIG(C), reuse);
	SeqMatMatMultAdd(alpha, CdamParMatIS(A), CdamParMatSG(B), 1.0, CdamParMatIG(C), MAT_REUSE);
	SeqMatMatMultAdd(alpha, CdamParMatIG(A), CdamParMatGG(B), 1.0, CdamParMatIG(C), MAT_REUSE);

	SeqMatMatMultAdd(alpha, CdamParMatSI(A), CdamParMatII(B), beta, CdamParMatSI(C), reuse);
	SeqMatMatMultAdd(alpha, CdamParMatSS(A), CdamParMatSI(B), 1.0, CdamParMatSI(C), MAT_REUSE);
	SeqMatMatMultAdd(alpha, CdamParMatSG(A), CdamParMatGI(B), 1.0, CdamParMatSI(C), MAT_REUSE);

	SeqMatMatMultAdd(alpha, CdamParMatSI(A), CdamParMatIS(B), beta, CdamParMatSS(C), reuse);
	SeqMatMatMultAdd(alpha, CdamParMatSS(A), CdamParMatSS(B), 1.0, CdamParMatSS(C), MAT_REUSE);
	SeqMatMatMultAdd(alpha, CdamParMatSG(A), CdamParMatGS(B), 1.0, CdamParMatSS(C), MAT_REUSE);

	SeqMatMatMultAdd(alpha, CdamParMatSI(A), CdamParMatIG(B), beta, CdamParMatSG(C), reuse);
	SeqMatMatMultAdd(alpha, CdamParMatSS(A), CdamParMatSG(B), 1.0, CdamParMatSG(C), MAT_REUSE);
	SeqMatMatMultAdd(alpha, CdamParMatSG(A), CdamParMatGG(B), 1.0, CdamParMatSG(C), MAT_REUSE);

	SeqMatMatMultAdd(alpha, CdamParMatGI(A), CdamParMatII(B), beta, CdamParMatGI(C), reuse);
	SeqMatMatMultAdd(alpha, CdamParMatGS(A), CdamParMatSI(B), 1.0, CdamParMatGI(C), MAT_REUSE);
	SeqMatMatMultAdd(alpha, CdamParMatGG(A), CdamParMatIG(B), 1.0, CdamParMatGI(C), MAT_REUSE);

	SeqMatMatMultAdd(alpha, CdamParMatGI(A), CdamParMatIS(B), beta, CdamParMatGS(C), reuse);
	SeqMatMatMultAdd(alpha, CdamParMatGS(A), CdamParMatSS(B), 1.0, CdamParMatGS(C), MAT_REUSE);
	SeqMatMatMultAdd(alpha, CdamParMatGG(A), CdamParMatSG(B), 1.0, CdamParMatGS(C), MAT_REUSE);

	SeqMatMatMultAdd(alpha, CdamParMatGI(A), CdamParMatIG(B), beta, CdamParMatGG(C), reuse);
	SeqMatMatMultAdd(alpha, CdamParMatGS(A), CdamParMatSG(B), 1.0, CdamParMatGG(C), MAT_REUSE);
	SeqMatMatMultAdd(alpha, CdamParMatGG(A), CdamParMatGG(B), 1.0, CdamParMatGG(C), MAT_REUSE);

}
void CdamParMatGetDiag(void* A, value_type* diag, index_type bs) {
	index_type nrow_exclusive = CdamParMatNumRowExclusive(A);	
	index_type nrow_shared = CdamParMatNumRowShared(A);
	SeqMatGetDiag(CdamParMatII(A), diag, bs);
	SeqMatGetDiag(CdamParMatSS(A), diag + bs * bs * nrow_exclusive, bs);
	if(CdamParMatAssemblyType(A) == MAT_DISASSEMBLED) {
		SeqMatGetDiag(CdamParMatGG(A), diag + bs * bs * (nrow_exclusive + nrow_shared), bs);
		CdamCommutor* commu = (CdamCommutor*)CdamParMatCommutor(A);
		CdamLayout* map = CdamParMatRowMap(A);
		CdamCommuForward(commu, diag, map, bs * bs * (int)sizeof(value_type));
		CdamCommuBackward(commu, diag, map, bs * bs * (int)sizeof(value_type));
	}
}
void CdamParMatAddElemValueBatched(void* A,
																	 index_type nelem, index_type nshl, index_type* ien,
																	 value_type* value, Arena scratch) {

	index_type nrow_offset[4] = {0, 0, 0, 0};
	CdamLayout* map = CdamParMatRowMap(A);
	nrow_offset[0] = 0;
	nrow_offset[1] = nrow_offset[0] + CdamLayoutNumExlusive(map);
	nrow_offset[2] = nrow_offset[1] + CdamLayoutNumShared(map);
	nrow_offset[3] = nrow_offset[2] + CdamLayoutNumGhosted(map);
	index_type* index = ArenaAlloc(sizeof(index_type), nelem * nshl, &scratch, 0);

	index_type nr_exclusive, nr_shared, nr_ghosted;
	/* Collect the indices of ien that are in the exclusive, shared, and ghosted rows */
	SelectIndexByRange(nelem * nshl, ien, index, nrow_offset[0], nrow_offset[1]);
	nr_exclusive = CountIndexByRange(nelem * nshl, ien, nrow_offset[0], nrow_offset[1]);
	/* Collect the indices of ien that are in the shared rows */
	SelectIndexByRange(nelem * nshl, ien, index + nr_exclusive, nrow_offset[1], nrow_offset[2]);
	nr_shared = CountIndexByRange(nelem * nshl, ien, nrow_offset[1], nrow_offset[2]);
	/* Collect the indices of ien that are in the ghosted rows */
	SelectIndexByRange(nelem * nshl, ien, index + nr_exclusive + nr_shared, nrow_offset[2], nrow_offset[3]);
	nr_ghosted = CountIndexByRange(nelem * nshl, ien, nrow_offset[2], nrow_offset[3]);

	/* Add the values by submatrices */
	SeqMatAddElemValueBatched(CdamParMatII(A), nelem, nshl, ien, value,
														nr_exclusive, index, nr_exclusive, index, scratch);
	SeqMatAddElemValueBatched(CdamParMatIS(A), nelem, nshl, ien, value,
														nr_exclusive, index, nr_shared, index + nr_exclusive, scratch);
	SeqMatAddElemValueBatched(CdamParMatIG(A), nelem, nshl, ien, value,
														nr_exclusive, index, nr_ghosted, index + nr_exclusive + nr_shared, scratch);

	SeqMatAddElemValueBatched(CdamParMatSI(A), nelem, nshl, ien, value,
														nr_shared, index + nr_exclusive, nr_exclusive, index, scratch);
	SeqMatAddElemValueBatched(CdamParMatSS(A), nelem, nshl, ien, value,
														nr_shared, index + nr_exclusive, nr_shared, index + nr_exclusive, scratch);
	SeqMatAddElemValueBatched(CdamParMatSG(A), nelem, nshl, ien, value,
														nr_shared, index + nr_exclusive, nr_ghosted, index + nr_exclusive + nr_shared, scratch);

	SeqMatAddElemValueBatched(CdamParMatGI(A), nelem, nshl, ien, value,
														nr_ghosted, index + nr_exclusive + nr_shared, nr_exclusive, index, scratch);
	SeqMatAddElemValueBatched(CdamParMatGS(A), nelem, nshl, ien, value,
														nr_ghosted, index + nr_exclusive + nr_shared, nr_shared, index + nr_exclusive, scratch);
	SeqMatAddElemValueBatched(CdamParMatGG(A), nelem, nshl, ien, value,
														nr_ghosted, index + nr_exclusive + nr_shared, nr_ghosted, index + nr_exclusive + nr_shared, scratch);

}


__END_DECLS__
