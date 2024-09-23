#ifndef __PARALLEL_MATRIX_H__
#define __PARALLEL_MATRIX_H__

#include "common.h"

__BEGIN_DECLS__

struct CdamParMatOp {
	void (*setup)(void* A, void* ctx);
	void (*destroy)(void* A);

	void (*zero)(void* A);
	void (*zere_row)(void* A, index_type row, index_type* rows, index_shift shift, value_type diag);
	void (*get_submat)(void* A, index_type nr, index_type* rows, index_type nc, index_type* cols, void* B);

	void (*copy)(void* A, void* B);
	void (*transpose)(void* A, void* B);

	void (*multadd)(value_type alpha, void* A, value_type* x, value_type beta, value_type* y);
	void (*matmultadd)(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);
	void (*get_diag)(void* A, value_type* diag, index_type bs);
	void (*add_elem_value_batched)(void* A, index_type batch_size, index_type* ien, index_type nshl,
																 index_type block_row, index_type block_col,
																 value_type* value, index_type ldv, index_type stride);
};
typedef struct CdamParMatOp CdamParMatOp;

struct CdamParMat {
	MPI_Comm comm;
	index_type global_nrows;
	index_type global_ncols;

	index_type row_range[2];
	index_type row_count[3];

	index_type col_range[2];
	index_type col_count[3];
	
	index_type* row_map;
	index_type* col_map;

	void* diag;
	void* offd;
	
	CdamParMatOp op[1];
};

typedef struct CdamParMat CdamParMat;

#define CdamParMatNumRowGlobal(A) (((CdamParMat*)(A))->global_nrows)
#define CdamParMatNumColGlobal(A) (((CdamParMat*)(A))->global_ncols)
#define CdamParMatRowBegin(A) (((CdamParMat*)(A))->row_range[0])
#define CdamParMatRowEnd(A) (((CdamParMat*)(A))->row_range[1])
#define CdamParMatNumRowExclusive(A) (((CdamParMat*)(A))->row_count[0])
#define CdamParMatNumRowShared(A) (((CdamParMat*)(A))->row_count[1])
#define CdamParMatNumRowGhosted(A) (((CdamParMat*)(A))->row_count[2])
#define CdamParMatColBegin(A) (((CdamParMat*)(A))->col_range[0])
#define CdamParMatColEnd(A) (((CdamParMat*)(A))->col_range[1])
#define CdamParMatNumColExclusive(A) (((CdamParMat*)(A))->col_count[0])
#define CdamParMatNumColShared(A) (((CdamParMat*)(A))->col_count[1])
#define CdamParMatNumColGhosted(A) (((CdamParMat*)(A))->col_count[2])
#define CdamParMatRowMap(A) (((CdamParMat*)(A))->row_map)
#define CdamParMatColMap(A) (((CdamParMat*)(A))->col_map)
#define CdamParMatDiag(A) (((CdamParMat*)(A))->diag)
#define CdamParMatOffd(A) (((CdamParMat*)(A))->offd)



void CdamParMatCreate(MPI_Comm comm, void** A);
void CdamParMatDestroy(void* A);
void CdamParMatSetup(void* A);

void CdamParMatZero(void* A);
void CdamParMatZeroRow(void* A, index_type row, index_type* rows, index_shift shift, value_type diag);
void CdamParMatGetSubmat(void* A, index_type nr, index_type* rows, index_type nc, index_type* cols, void* B);

void CdamParMatCopy(void* A, void* B);
void CdamParMatTranspose(void* A, void* B);
void CdamParMatMultAdd(value_type alpha, void* A, value_type* x, value_type beta, value_type* y);
void CdamParMatMatMultAdd(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);
void CdamParMatGetDiag(void* A, value_type* diag, index_type bs);
void CdamParMatAddElemValueBatched(void* A, index_type batch_size, index_type* ien, index_type nshl,
																 index_type block_row, index_type block_col,
																 value_type* value, index_type ldv, index_type stride);


__END_DECLS__

#endif /* __PARALLEL_MATRIX_H__ */
