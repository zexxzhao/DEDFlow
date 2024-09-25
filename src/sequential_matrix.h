#ifndef __SEQUENTIAL_MATRIX_H__
#define __SEQUENTIAL_MATRIX_H__

#include "common.h"
#include "csr.h"
#include "layout.h"
#include "matrix_util.h"

__BEGIN_DECLS__

struct SeqMatOp {
	void (*setup)(void* A);
	void (*destroy)(void* A);

	void (*zero)(void* A);
	void (*zere_row)(void* A, index_type row, index_type* rows, index_type shift, value_type diag);
	void (*get_submat)(void* A, index_type nrow, index_type* row, index_type ncol, index_type* col, void* B);

	// void (*copy)(void* A, void* B);
	void (*transpose)(void* A);

	void (*multadd)(value_type alpha, void* A, value_type* x, value_type beta, value_type* y);
	void (*multransposeadd)(value_type alpha, void* A, value_type* x, value_type beta, value_type* y);
	void (*matmultadd)(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);
	void (*mattransposemultadd)(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);
	void (*get_diag)(void* A, value_type* diag, index_type bs);
	void (*add_elem_value_batched)(void* A, index_type nelem, index_type nshl, index_type* ien,
																 index_type nr, index_type* row, index_type nc, index_type* col,
																 value_type* value, Arena scratch);
};

struct SeqMat {
	MatType type;
	CdamLayout* rmap;
	CdamLayout* cmap;
	index_type size[2];
	void* data;
	struct SeqMatOp op[1];
};

#define SeqMatType(A) (((SeqMat*)A)->type)
#define SeqMatRowLayout(A) (((SeqMat*)A)->rmap)
#define SeqMatColLayout(A) (((SeqMat*)A)->cmap)
#define SeqMatNumRow(A) (((SeqMat*)A)->size[0])
#define SeqMatNumCol(A) (((SeqMat*)A)->size[1])
#define SeqMatData(A) (((SeqMat*)A)->data)
#define SeqMatAsType(A, T) ((T*)((SeqMat*)A)->data)

struct SeqMatDense {
	value_type* data;
};

struct SeqMatCSR {
	CSRAttr* spy;

	value_type* data;

	SPMatDesc descr;
	index_type buffer_size;
	void* buffer;
};

struct SeqMatSlicedELL {
	index_type block_size;
	index_type nblocks;
	index_type* col_ptr;
	index_type* row_ind;
	value_type* val;
};

#define SEQMAT_NESTED_MAX_SEG (16)
struct SeqMatNested {
	/* The submatrices in a nested matrix must have contiguous row/col indices */
	/* Row dof segment */
	index_type num_row_seg;
	index_type row_seg_offset[SEQMAT_NESTED_MAX_SEG + 1];
	index_type row_seg_size[SEQMAT_NESTED_MAX_SEG];

	/* Column dof segment */
	index_type num_col_seg;
	index_type col_seg_offset[SEQMAT_NESTED_MAX_SEG + 1];
	index_type col_seg_size[SEQMAT_NESTED_MAX_SEG];

	/* Submatrices */
	struct SeqMat* submatrices[SEQMAT_NESTED_MAX_SEG][SEQMAT_NESTED_MAX_SEG];
};

struct SeqMatVirtual {
	struct SeqMat* parent;
	index_type *row;
	index_type *col;
	value_type *prolonged_input;
	value_type *prolonged_output;
};

typedef struct SeqMat SeqMat;
typedef struct SeqMatDense SeqMatDense;
typedef struct SeqMatCSR SeqMatCSR;
typedef struct SeqMatSlicedELL SeqMatSlicedELL; /* Not implemented */
typedef struct SeqMatNested SeqMatNested; /* Not implemented */
typedef struct SeqMatVirtual SeqMatVirtual;

void SeqMatCreate(MatType type, index_type nrow, index_type ncol, void** A);
void SeqMatSetup(void* A);
void SeqMatDestroy(void* A);

void SeqMatZero(void* A);
void SeqMatZeroRow(void* A, index_type row, index_type* rows, index_type shift, value_type diag);

void SeqMatGetSubmat(void* A, index_type nrow, index_type* row, index_type ncol, index_type* col, void* B);

void SeqMatTranspose(void* A);

void SeqMatMultAdd(value_type alpha, void* A, value_type* x, value_type beta, value_type* y);
void SeqMatMulTransposeAdd(value_type alpha, void* A, value_type* x, value_type beta, value_type* y);
void SeqMatMatMultAdd(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);
void SeqMatMatTransposeMultAdd(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);

void SeqMatGetDiag(void* A, value_type* diag, index_type bs);
void SeqMatAddElemValueBatched(void* A, index_type nelem, index_type nshl, index_type* ien,
																index_type nr, index_type* row, index_type nc, index_type* col,
																value_type* value, Arena scratch);


__END_DECLS__
#endif /* __SEQUENTIAL_MATRIX_H__ */
