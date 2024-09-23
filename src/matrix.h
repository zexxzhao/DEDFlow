#ifndef __MATRIX_H__
#define __MATRIX_H__


#include "csr.h"

__BEGIN_DECLS__

typedef struct CdamMatOp CdamMatOp;

typedef enum {
	MAT_TYPE_NONE = 0,
	MAT_TYPE_DENSE = 1, /* Dense matrix */
	MAT_TYPE_CSR = 2,   /* Compressed Sparse Row matrix */
	MAT_TYPE_SELL = 4,  /* Sliced ELL matrix */
	MAT_TYPE_FS = 8,    /* Field Splitting matrix */
	MAT_TYPE_CUSTOM = 16 /* Custom matrix */
} CdamMatType;

typedef enum {
	MAT_INITIAL = 0,
	MAT_REUSE = 1
}	CdamMatReuse;

typedef enum {
	MAT_ROW_MAJOR = 0,
	MAT_COL_MAJOR = 1
} CdamMatOrder;


struct CdamMatOp {
	void (*setup)(CdamMat *matrix);

	void (*get_submat)(CdamMat *matrix, index_type nrow, index_type *row,
										 index_type ncol, index_type *col, CdamMat *submat);
	void (*zero)(CdamMat *matrix);
	void (*zero_row)(CdamMat *matrix, index_type , const index_type* row, index_type shift, value_type diag);

	void (*multadd)(value_type alpha, CdamMat* A, value_type* x, value_type beta, value_type* y);
	void (*matmultadd)(value_type alpha, CdamMat* A, CdamMat* B, value_type beta, CdamMat* C, MatReuse reuse);

	void (*get_diag)(CdamMat *matrix, value_type *diag, index_type bs);

	void (*set_values_coo)(CdamMat* matrix, value_type alpha, index_type n, const index_type* row, const index_type* col, const value_type* val, value_type beta);
	void (*set_values_ind)(CdamMat* matrix, value_type alpha, index_type n, const index_type* ind, const value_type* val, value_type beta);

	void (*add_elem_value_batched)(CdamMat* matrix, index_type nshl, \
																 index_type batch_size, const index_type* batch_ptr, const index_type* ien, \
																 const value_type* val, const index_type* mask);
	void (*add_elem_value_blocked_batched)(CdamMat* matrix, index_type nshl, \
																				 index_type batch_size, const index_type* batch_ptr, const index_type* ien, \
																				 index_type block_row_size, index_type block_col_size, \
																				 const value_type* val, int lda, int stride, const index_type* mask);
	void (*add_value_batched)(CdamMat* matrix,
														index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
														const value_type* A);
	void (*add_value_blocked_batched)(CdamMat* matrix,
																		index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
																		index_type block_row, index_type block_col,
																		const value_type* A, int lda, int stride);

	void (*destroy)(CdamMat *matrix);
};

struct CdamMat {
	CdamMatType type;
	index_type row_range[2];
	index_type row_count[3];
	index_type col_range[4];
	index_type col_count[3];
	void *data;
#ifdef CDAM_USE_CUDA
	cudaStream_t stream_ref;
#endif
	CdamMatOp op[1];
};

#define CdamMatType(A) ((A)->type)
#define CdamMatRowBegin(A) ((A)->row_range[0])
#define CdamMatRowEnd(A) ((A)->row_range[1])
#define CdamMatNumExclusiveRow(A) ((A)->row_count[0])
#define CdamMatNumSharedRow(A) ((A)->row_count[1])
#define CdamMatNumGhostedRow(A) ((A)->row_count[2])

#define CdamMatColBegin(A) ((A)->col_range[0])
#define CdamMatColEnd(A) ((A)->col_range[1])
#define CdamMatNumExclusiveCol(A) ((A)->col_count[0])
#define CdamMatNumSharedCol(A) ((A)->col_count[1])
#define CdamMatNumGhostedCol(A) ((A)->col_count[2])

#define CdamMatImpl(A) ((A)->data)

#define AsCdamMatType(A, type) ((type*)CdamMatImpl(A))

struct CdamMatDense {
	MatOrder order;
	value_type *val;
	index_type ld;
};

struct CdamMatCSR {
	b32 external_attr;
	const CSRAttr *attr;
	value_type *val;
	SPMatDesc d_descr;

	index_type buffer_size;
	void* buffer;
};

struct CdamMatBSR {
	CSRAttr *attr;
	index_type block_size[2];
	value_type *val;
	
	SPMatDesc d_descr;
	index_type buffer_size;
	void* buffer;
};

#define CdamMatCSRAttr(matrix) ((matrix)->attr)
#define CdamMatCSRRowPtr(matrix) (CSRAttrRowPtr(CdamMatCSRAttr(matrix)))
#define CdamMatCSRColInd(matrix) (CSRAttrColInd(CdamMatCSRAttr(matrix)))
#define CdamMatCSRData(matrix) ((matrix)->data)
#define CdamMatCSRNumRow(matrix) (CSRAttrNumRow(CdamMatCSRAttr(matrix)))
#define CdamMatCSRNumCol(matrix) (CSRAttrNumCol(CdamMatCSRAttr(matrix)))
#define CdamMatCSRNNZ(matrix) (CSRAttrNNZ(CdamMatCSRAttr(matrix)))
#define CdamMatCSRDescr(matrix) ((matrix)->descr)


struct CdamMatFS {
	index_type n_offset;
	index_type *offset;
	index_type *d_offset;

#ifdef CDAM_USE_CUDA
	cudaStream_t* stream;
#endif

	const CSRAttr* spy1x1;
	value_type** d_matval;
	CdamMat** mat;
};


/* API for CdamMat */
CdamMat* CdamMatCreateTypeCSR(const CSRAttr* attr, void*);
CdamMat* CdamMatCreateTypeFS(index_type n_offset, const index_type* offset, void*);
void CdamMatDestroy(CdamMat *matrix);
void CdamMatSetup(CdamMat *matrix);
void CdamMatZero(CdamMat *matrix);
void CdamMatZeroRow(CdamMat *matrix, index_type n, const index_type* row, index_type shift, value_type diag);
void CdamMatAMVPBY(CdamMat* A, value_type alpha, value_type* x, value_type beta, value_type* y);
void CdamMatMatVec(CdamMat *matrix,  value_type *x, value_type *y);

void CdamMatSubMatVec(CdamMat *matrix, index_type i, index_type* ix, index_type j, index_type* jx, value_type* x, value_type* y);

void CdamMatGetDiag(CdamMat *matrix, value_type *diag, index_type bs);

void CdamMatSetValuesCOO(CdamMat* matrix, value_type alpha, index_type n, const index_type* row, const index_type* col, const value_type* val, value_type beta);
void CdamMatSetValuesInd(CdamMat* matrix, value_type alpha, index_type n, const index_type* ind, const value_type* val, value_type beta);

void CdamMatAddElemValueBatched(CdamMat* matrix, index_type nshl,
															 index_type num_batch, const index_type* batch_ptr, const index_type* ien,
															 const value_type* val, const index_type* mask);
void CdamMatAddElemValueBlockedBatched(CdamMat* matrix, index_type nshl,
																			index_type num_batch, const index_type* batch_ptr, const index_type* ien,
																			index_type block_row_size, index_type block_col_size,
																			const value_type* val, int lda, int stride, const index_type* mask);

void CdamMatAddValueBatched(CdamMat* matrix,
													 index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
													 const value_type* A);
void CdamMatAddValueBlockedBatched(CdamMat* matrix,
																	index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
																	index_type block_row_size, index_type block_col_size,
																	const value_type* A, int lda, int stride);


/* API for Type CSR */
CdamMatCSR* CdamMatCSRCreate(const CSRAttr *attr, void*);
void CdamMatCSRDestroy(CdamMat *matrix);


/* API for Type FS */
CdamMatFS* CdamMatFSCreate(index_type n_offset, const index_type *offset, void*);
void CdamMatFSDestroy(CdamMat *matrix);

void CdamMatCreate(CdamMat **mat, MatType type);

void CdamMatDestroy(CdamMat *mat);

void CdamMatGetSubMat(CdamMat *matrix, index_type nrow, index_type *row, index_type ncol, index_type *col, CdamMat *submat);




typedef CdamMat CdamMat;

__END_DECLS__

#endif /* __MATRIX_H__ */
