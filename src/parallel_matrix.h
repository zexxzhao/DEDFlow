#ifndef __PARALLEL_MATRIX_H__
#define __PARALLEL_MATRIX_H__

#include "common.h"

__BEGIN_DECLS__

struct CdamParMatOp {
	void (*setup)(void* A);
	void (*destroy)(void* A);

	void (*zero)(void* A);
	void (*zere_row)(void* A, index_type row, index_type* rows, index_shift shift, value_type diag);
	void (*multadd)(value_type alpha, void* A, value_type* x, value_type beta, value_type* y);
	void (*matmultall)(value_type alpha, void* A, void* B, value_type beta, void* C, MatReuse reuse);
	void (*get_diag)(void* A, value_type* diag, index_type bs);
	void (*add_value_batched)(void* A, index_type batch_size, index_type* ien, index_type nshl,
			                      index_type block_row, index_type block_col,
														value_type* value, index_type ldv, index_type stride);
};
typedef struct CdamParMatOp CdamParMatOp;

struct CdamParMat {
	MPI_Comm comm;
	index_type global_nrows;
	index_type global_ncols;
	index_type global_nnz;

	index_type row_range[2];
	index_type row_count[3];

	index_type col_range[2];
	index_type col_count[3];
	
	void* diag;
	void* offd;
	
	CdamParMatOp op[1];
};

typedef struct CdamParMat CdamParMat;


__END_DECLS__

#endif /* __PARALLEL_MATRIX_H__ */
