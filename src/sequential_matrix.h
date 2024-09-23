#ifndef __SEQUENTIAL_MATRIX_H__
#define __SEQUENTIAL_MATRIX_H__

#include "common.h"

__BEGIN_DECLS__

typedef enum {
	MAT_TYPE_NONE = 0,
	MAT_TYPE_DENSE = 1, /* Dense matrix */
	MAT_TYPE_CSR = 2,   /* Compressed Sparse Row matrix */
	MAT_TYPE_SELL = 4,   /* Slice ELL matrix */
	MAT_TYPE_FS = 8,    /* Field Splitting matrix */
	MAT_TYPE_CUSTOM = 16 /* Custom matrix */
} MatType;

typedef enum {
	MAT_INITIAL = 0,
	MAT_REUSE = 1
}	MatReuse;

typedef enum {
	MAT_ROW_MAJOR = 0,
	MAT_COL_MAJOR = 1
} MatOrder;

struct SeqMatOp {
	void (*setup)(void* A);
	void (*destroy)(void* A);

	void (*zero)(void* A);
	void (*zere_row)(void* A, index_type row, index_type* rows, index_shift shift, value_type diag);

	void (*copy)(void* A, void* B);
	void (*transpose)(void* A, void* B);

	void (*multadd)(value_type alpha, void* A, value_type* x, value_type beta, value_type* y);
	void (*matmultall)(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);
	void (*get_diag)(void* A, value_type* diag, index_type bs);
	void (*add_value_batched)(void* A, index_type batch_size, index_type* ien, index_type nshl,
			                      index_type block_row, index_type block_col,
														value_type* value, index_type ldv, index_type stride);
};

struct SeqMat {
	MatType type;
	index_type size[2];
	void* data;
	SeqMatOp op[1];
};

#define SeqMatType(A) (((SeqMat*)A)->type)
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

// struct SeqMatSlicedELL {
// 	index_type block_size;
// 	index_type nblocks;
// 	index_type* col_ptr;
// 	index_type* row_ind;
// 	value_type* val;
// };

typedef struct SeqMat SeqMat;
typedef struct SeqMatDense SeqMatDense;
typedef struct SeqMatCSR SeqMatCSR;


__END_DECLS__
#endif /* __SEQUENTIAL_MATRIX_H__ */
