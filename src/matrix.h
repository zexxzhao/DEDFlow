#ifndef __MATRIX_H__
#define __MATRIX_H__


#include "csr.h"

__BEGIN_DECLS__

typedef struct MatrixOp MatrixOp;

enum MatType{
	MAT_TYPE_NONE = 0,
	MAT_TYPE_DENSE = 1, /* Dense matrix */
	MAT_TYPE_CSR = 2,   /* Compressed Sparse Row matrix */
	MAT_TYPE_FS = 4,    /* Field Splitting matrix */
	MAT_TYPE_CUSTOM = 8 /* Custom matrix */
};

typedef enum MatType MatType;

typedef struct Matrix Matrix;
typedef struct MatrixCSR MatrixCSR;
typedef struct MatrixFS MatrixFS;

struct MatrixOp {
	void (*setup)(Matrix *matrix);

	void (*zero)(Matrix *matrix);
	void (*zero_row)(Matrix *matrix, index_type , const index_type* row, index_type shift, value_type diag);

	void (*amvpby)(Matrix* A,  value_type alpha, value_type* x, value_type beta, value_type* y);
	void (*amvpby_mask)(Matrix* A,  value_type alpha, value_type* x, value_type beta, value_type* y,\
											value_type* left_mask,  value_type* right_mask);
	void (*matvec)(Matrix *matrix,  value_type *x, value_type *y);
	void (*matvec_mask)(Matrix *matrix,  value_type *x, value_type *y,\
											value_type* left_mask,  value_type *right_mask);

	void (*get_diag)(Matrix *matrix, value_type *diag, index_type bs);

	void (*set_values_coo)(Matrix* matrix, value_type alpha, index_type n, const index_type* row, const index_type* col, const value_type* val, value_type beta);
	void (*set_values_ind)(Matrix* matrix, value_type alpha, index_type n, const index_type* ind, const value_type* val, value_type beta);

	void (*add_elem_value_batched)(Matrix* matrix, index_type nshl, \
																 index_type batch_size, const index_type* batch_ptr, const index_type* ien, \
																 const value_type* val, const index_type* mask);
	void (*add_elem_value_blocked_batched)(Matrix* matrix, index_type nshl, \
																				 index_type batch_size, const index_type* batch_ptr, const index_type* ien, \
																				 index_type block_row_size, index_type block_col_size, \
																				 const value_type* val, int lda, int stride, const index_type* mask);
	void (*add_value_batched)(Matrix* matrix,
														index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
														const value_type* A);
	void (*add_value_blocked_batched)(Matrix* matrix,
																		index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
																		index_type block_row, index_type block_col,
																		const value_type* A, int lda, int stride);

	void (*destroy)(Matrix *matrix);
};

struct Matrix {
	index_type size[2];
	MatType type;
	void *data;
	cudaStream_t stream_ref;
	MatrixOp op[1];
};

#define MatrixNumRow(A) ((A)->size[0])
#define MatrixNumCol(A) ((A)->size[1])
#define MatrixType(A) ((A)->type)

struct MatrixCSR {
	b32 external_attr;
	const CSRAttr *attr;
	value_type *val;
	cusparseSpMatDescr_t descr;
	index_type buffer_size;
	void* buffer;
};

#define MatrixCSRAttr(matrix) ((matrix)->attr)
#define MatrixCSRRowPtr(matrix) (CSRAttrRowPtr(MatrixCSRAttr(matrix)))
#define MatrixCSRColInd(matrix) (CSRAttrColInd(MatrixCSRAttr(matrix)))
#define MatrixCSRData(matrix) ((matrix)->data)
#define MatrixCSRNumRow(matrix) (CSRAttrNumRow(MatrixCSRAttr(matrix)))
#define MatrixCSRNumCol(matrix) (CSRAttrNumCol(MatrixCSRAttr(matrix)))
#define MatrixCSRNNZ(matrix) (CSRAttrNNZ(MatrixCSRAttr(matrix)))
#define MatrixCSRDescr(matrix) ((matrix)->descr)

struct MatrixFS {
	index_type n_offset;
	index_type *offset;
	index_type *d_offset;

	cudaStream_t* stream;

	const CSRAttr* spy1x1;
	value_type** d_matval;
	Matrix** mat;
};


/* API for Matrix */
Matrix* MatrixCreateTypeCSR(const CSRAttr* attr, void*);
Matrix* MatrixCreateTypeFS(index_type n_offset, const index_type* offset, void*);
void MatrixDestroy(Matrix *matrix);
void MatrixSetup(Matrix *matrix);
void MatrixZero(Matrix *matrix);
void MatrixZeroRow(Matrix *matrix, index_type n, const index_type* row, index_type shift, value_type diag);
void MatrixAMVPBY(Matrix* A, value_type alpha, value_type* x, value_type beta, value_type* y);
void MatrixAMVPBYWithMask(Matrix* A,  value_type alpha, value_type* x, value_type beta, value_type* y,\
													 value_type* left_mask,  value_type* right_mask);
void MatrixMatVec(Matrix *matrix,  value_type *x, value_type *y);
void MatrixMatVecWithMask(Matrix *matrix,  value_type *x, value_type *y,  value_type *left_mask, value_type* right_mask);
void MatrixGetDiag(Matrix *matrix, value_type *diag, index_type bs);

void MatrixSetValuesCOO(Matrix* matrix, value_type alpha, index_type n, const index_type* row, const index_type* col, const value_type* val, value_type beta);
void MatrixSetValuesInd(Matrix* matrix, value_type alpha, index_type n, const index_type* ind, const value_type* val, value_type beta);

void MatrixAddElemValueBatched(Matrix* matrix, index_type nshl,
															 index_type num_batch, const index_type* batch_ptr, const index_type* ien,
															 const value_type* val, const index_type* mask);
void MatrixAddElemValueBlockedBatched(Matrix* matrix, index_type nshl,
																			index_type num_batch, const index_type* batch_ptr, const index_type* ien,
																			index_type block_row_size, index_type block_col_size,
																			const value_type* val, int lda, int stride, const index_type* mask);

void MatrixAddValueBatched(Matrix* matrix,
													 index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
													 const value_type* A);
void MatrixAddValueBlockedBatched(Matrix* matrix,
																	index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
																	index_type block_row_size, index_type block_col_size,
																	const value_type* A, int lda, int stride);


/* API for Type CSR */
MatrixCSR* MatrixCSRCreate(const CSRAttr *attr, void*);
void MatrixCSRDestroy(Matrix *matrix);


/* API for Type FS */
MatrixFS* MatrixFSCreate(index_type n_offset, const index_type *offset, void*);
void MatrixFSDestroy(Matrix *matrix);


typedef Matrix CdamMat;

__END_DECLS__

#endif /* __MATRIX_H__ */
