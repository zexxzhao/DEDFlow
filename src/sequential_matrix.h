#ifndef __SEQUENTIAL_MATRIX_H__
#define __SEQUENTIAL_MATRIX_H__

#include "common.h"
#include "matrix_util.h"

__BEGIN_DECLS__

struct SeqMatOp {
	void (*setup)(void* A);
	void (*destroy)(void* A);

	void (*zero)(void* A);
	void (*zere_row)(void* A, index_type row, index_type* rows, index_shift shift, value_type diag);
	void (*get_submat)(void* A, index_type nr, index_type* rows, index_type nc, index_type* cols, void* B, void** auxiliary_data);

	void (*copy)(void* A, void* B);
	void (*transpose)(void* A, void* B);

	void (*multadd)(value_type alpha, void* A, value_type* x, value_type beta, value_type* y);
	void (*matmultadd)(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);
	void (*get_diag)(void* A, value_type* diag, index_type bs);
	void (*add_elem_value_batched)(void* A, index_type batch_size, index_type* ien, index_type nshl,
																 index_type block_row, index_type block_col,
																 value_type* value, index_type ldv, index_type stride);
};

struct SeqMat {
	MatType type;
	MatStorageMethod rmap_storage;
	MatStorageMethod cmap_storage;
	index_type size[2];
	void* data;
	SeqMatOp op[1];
};

#define SeqMatType(A) (((SeqMat*)A)->type)
#define SeqMatRowStorageMethod(A) (((SeqMat*)A)->rmap_storage)
#define SeqMatColStorageMethod(A) (((SeqMat*)A)->cmap_storage)
#define SeqMatNumRow(A) (((SeqMat*)A)->size[0])
#define SeqMatNumCol(A) (((SeqMat*)A)->size[1])
#define SeqMatData(A) (((SeqMat*)A)->data)
#define SeqMatAsType(A, T) ((T*)((SeqMat*)A)->data)

struct SeqMatDense {
	MatOrder order;
	value_type* data;
};

struct SeqMatCSR {
	CSRAttr* spy;

	value_type* data;

	SpMatDescr descr;
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

typedef struct SeqMat SeqMat;
typedef struct SeqMatDense SeqMatDense;
typedef struct SeqMatCSR SeqMatCSR;

void SeqMatCreate(MatType type, index_type nrow, index_type ncol, void** A);
void SeqMatSetup(void* A);
void SeqMatDestroy(void* A);

void SeqMatZero(void* A);
void SeqMatZeroRow(void* A, index_type row, index_type* rows, index_shift shift, value_type diag);

void SeqMatGetSubmat(void* A, index_type nr, index_type* rows, index_type nc, index_type* cols, void* B, void** auxiliary_data);

void SeqMatCopy(void* A, void* B);
void SeqMatTranspose(void* A, void* B);

void SeqMatMultAdd(value_type alpha, void* A, value_type* x, value_type beta, value_type* y);
void SeqMatMatMultAdd(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);

void SeqMatGetDiag(void* A, value_type* diag, index_type bs);
void SeqMatAddElemValueBatched(void* A, index_type batch_size, index_type* ien, index_type nshl,
															 index_type block_row, index_type block_col,
															 value_type* value, index_type ldv, index_type stride);


__END_DECLS__
#endif /* __SEQUENTIAL_MATRIX_H__ */
