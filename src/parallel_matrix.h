#ifndef __PARALLEL_MATRIX_H__
#define __PARALLEL_MATRIX_H__

#include <mpi.h>
#include "common.h"
#include "matrix_util.h"

__BEGIN_DECLS__

struct CdamParMatOp {
	void (*setup)(void* A);
	void (*destroy)(void* A);

	void (*zero)(void* A);
	void (*zere_row)(void* A, index_type row, index_type* rows, value_type diag);
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
	MatAssemblyType assembly_type;
	index_type global_nrows;
	index_type global_ncols;

	index_type row_range[2];
	index_type row_count[3];

	index_type col_range[2];
	index_type col_count[3];
	
	CdamLayout* row_map;
	CdamLayout* col_map;

	void* commutor;
	void* submat[3][3];
	
	CdamParMatOp op[1];
};

typedef struct CdamParMat CdamParMat;
#define CdamParMatComm(A) (((CdamParMat*)(A))->comm)
#define CdamParMatAssemblyType(A) (((CdamParMat*)(A))->assembly_type)
#define CdamParMatNumRowGlobal(A) (((CdamParMat*)(A))->global_nrows)
#define CdamParMatNumColGlobal(A) (((CdamParMat*)(A))->global_ncols)
#define CdamParMatRowBegin(A) (((CdamParMat*)(A))->row_range[0])
#define CdamParMatRowEnd(A) (((CdamParMat*)(A))->row_range[1])
#define CdamParMatNumRowExclusive(A) (((CdamParMat*)(A))->row_count[0])
#define CdamParMatNumRowShared(A) (((CdamParMat*)(A))->row_count[1])
#define CdamParMatNumRowGhosted(A) (((CdamParMat*)(A))->row_count[2])
#define CdamParMatNumRowAll(A) (CdamParMatNumRowExclusive(A) + CdamParMatNumRowShared(A) + CdamParMatNumRowGhosted(A))
#define CdamParMatNumRowOwned(A) (CdamParMatNumRowExclusive(A) + CdamParMatNumRowShared(A))
#define CdamParMatColBegin(A) (((CdamParMat*)(A))->col_range[0])
#define CdamParMatColEnd(A) (((CdamParMat*)(A))->col_range[1])
#define CdamParMatNumColExclusive(A) (((CdamParMat*)(A))->col_count[0])
#define CdamParMatNumColShared(A) (((CdamParMat*)(A))->col_count[1])
#define CdamParMatNumColGhosted(A) (((CdamParMat*)(A))->col_count[2])
#define CdamParMatNumColAll(A) (CdamParMatNumColExclusive(A) + CdamParMatNumColShared(A) + CdamParMatNumColGhosted(A))
#define CdamParMatNumColOwned(A) (CdamParMatNumColExclusive(A) + CdamParMatNumColShared(A))
#define CdamParMatRowMap(A) (((CdamParMat*)(A))->row_map)
#define CdamParMatColMap(A) (((CdamParMat*)(A))->col_map)

#define CdamParMatCommutor(A) (((CdamParMat*)(A))->commutor)
#define CdamParMatII(A) (((CdamParMat*)(A))->submat[0][0])
#define CdamParMatIS(A) (((CdamParMat*)(A))->submat[0][1])
#define CdamParMatIG(A) (((CdamParMat*)(A))->submat[0][2])
#define CdamParMatSI(A) (((CdamParMat*)(A))->submat[1][1])
#define CdamParMatSS(A) (((CdamParMat*)(A))->submat[1][1])
#define CdamParMatSG(A) (((CdamParMat*)(A))->submat[1][2])
#define CdamParMatGI(A) (((CdamParMat*)(A))->submat[2][0])
#define CdamParMatGS(A) (((CdamParMat*)(A))->submat[2][1])
#define CdamParMatGG(A) (((CdamParMat*)(A))->submat[2][2])


void CdamParMatCreate(MPI_Comm comm, void** A);
void CdamParMatDestroy(void* A);
void CdamParMatSetup(void* A);

void CdamParMatZero(void* A);
void CdamParMatZeroRow(void* A, index_type row, index_type* rows, value_type diag);
void CdamParMatGetSubmat(void* A, index_type nr, index_type* rows, index_type nc, index_type* cols, void* B);

// void CdamParMatCopy(void* A, void* B);
void CdamParMatTranspose(void* A, void* B);
void CdamParMatMultAdd(value_type alpha, void* A, value_type* x, value_type beta, value_type* y);
void CdamParMatMatMultAdd(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);
void CdamParMatGetDiag(void* A, value_type* diag, index_type bs);
/** Add elementwise matrices to the matrix
 * Some restrictions apply:
 * value must be of size nelem*nshl*nshl*row_block_size*col_block_size,
 * where row_block_size is the block_size defined in the rmap, and col_block_size for cmap.
 * Even if not all elements are used, the size of value must be padded to the full size.
 * The blocks in value cannot be separated and added to different submatrices. For instance,
 * A block of 3x3 can be added to II monolithically, but it does not work if [0,1]x[0,1,2]
 * is added to II and [2]x[0,1,2] is added to IG. If separation is needed, consider splitting
 * into submatrices and adding them separately.
 * @param A matrix
 * @param nelem number of elements to add
 * @param nshl number of nodes per element
 * @param ien element node connectivity
 * @param value values to add
 */
void CdamParMatAddElemValueBatched(void* A,
																	 index_type nelem, index_type nshl, index_type* ien,
																	 value_type* value);


__END_DECLS__

#endif /* __PARALLEL_MATRIX_H__ */
