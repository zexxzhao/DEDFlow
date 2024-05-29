#ifndef __MATRIX_H__
#define __MATRIX_H__


#include "csr.h"

__BEGIN_DECLS__

typedef u32 index_type;
typedef f64 value_type;
typedef struct MatrixOp MatrixOp;

typedef enum {
	MAT_TYPE_NONE = 0,
	MAT_TYPE_DENSE = 1,
	MAT_TYPE_CSR = 2,
	MAT_TYPE_CUSTOM = 4,
	MAT_TYPE_NESTED = 8
} MatType;

typedef struct Matrix Matrix;
typedef struct MatrixCSR MatrixCSR;
typedef struct MatrixNested MatrixNested;

struct MatrixOp {
	void (*zero)(Matrix *matrix);
	void (*zero_row)(Matrix *matrix, index_type , const index_type* row, value_type diag);

	void (*amvpby)(value_type alpha, Matrix* A,  value_type* x, value_type beta, value_type* y);
	void (*amvpby_mask)(value_type alpha, Matrix* A,  value_type* x, value_type beta, value_type* y,\
											value_type* left_mask,  value_type* right_mask);
	void (*matvec)(Matrix *matrix,  value_type *x, value_type *y);
	void (*matvec_mask)(Matrix *matrix,  value_type *x, value_type *y,\
											value_type* left_mask,  value_type *right_mask);
	void (*get_diag)(Matrix *matrix, value_type *diag);

	void (*set_values_coo)(Matrix* matrix, value_type alpha, index_type n, const index_type* row, const index_type* col, const value_type* val, value_type beta);
	void (*set_values_ind)(Matrix* matrix, value_type alpha, index_type n, const index_type* ind, const value_type* val, value_type beta);

	void (*add_elem_value_batched)(Matrix* matrix, index_type nshl, \
																 index_type batch_size, const index_type* batch_ptr, const index_type* ien, \
																 const value_type* val);
	void (*add_elem_value_blocked_batched)(Matrix* matrix, index_type nshl, \
																				 index_type batch_size, const index_type* batch_ptr, const index_type* ien, \
																				 index_type block_row_size, index_type block_col_size, \
																				 const value_type* val, int lda, int stride);
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
	// MatrixOp *op, _op_private;
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

struct MatrixNested {
	index_type n_offset;
	const index_type *offset;
	index_type *d_offset;
	const CSRAttr* csr_auxiliary;
	Matrix* mat[0];
};


/* API for Matrix */
Matrix* MatrixCreateTypeCSR(const CSRAttr* attr);
Matrix* MatrixCreateTypeNested(index_type n_offset, const index_type* offset);
void MatrixDestroy(Matrix *matrix);
void MatrixZero(Matrix *matrix);
void MatrixZeroRow(Matrix *matrix, index_type n, const index_type* row, value_type diag);
void MatrixAMVPBY(value_type alpha, Matrix* A,  value_type* x, value_type beta, value_type* y);
void MatrixAMVPBYWithMask(value_type alpha, Matrix* A,  value_type* x, value_type beta, value_type* y,\
													 value_type* left_mask,  value_type* right_mask);
void MatrixMatVec(Matrix *matrix,  value_type *x, value_type *y);
void MatrixMatVecWithMask(Matrix *matrix,  value_type *x, value_type *y,  value_type *left_mask, value_type* right_mask);
void MatrixGetDiag(Matrix *matrix, value_type *diag);

void MatrixSetValuesCOO(Matrix* matrix, value_type alpha, index_type n, const index_type* row, const index_type* col, const value_type* val, value_type beta);
void MatrixSetValuesInd(Matrix* matrix, value_type alpha, index_type n, const index_type* ind, const value_type* val, value_type beta);

void MatrixAddElemValueBatched(Matrix* matrix, index_type nshl,
															 index_type num_batch, const index_type* batch_ptr, const index_type* ien,
															 const value_type* val);
void MatrixAddElemValueBlockedBatched(Matrix* matrix, index_type nshl,
																			index_type num_batch, const index_type* batch_ptr, const index_type* ien,
																			index_type block_row_size, index_type block_col_size,
																			const value_type* val, int lda, int stride);

void MatrixAddValueBatched(Matrix* matrix,
													 index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
													 const value_type* A);
void MatrixAddValueBlockedBatched(Matrix* matrix,
																	index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
																	index_type block_row_size, index_type block_col_size,
																	const value_type* A, int lda, int stride);


/* API for Type CSR */
MatrixCSR* MatrixCSRCreate(const CSRAttr *attr);
void MatrixCSRDestroy(Matrix *matrix);


/* API for Type Nested */
MatrixNested* MatrixNestedCreate(index_type n_offset, const index_type *offset);
void MatrixNestedDestroy(Matrix *matrix);


__END_DECLS__

#endif /* __MATRIX_H__ */
