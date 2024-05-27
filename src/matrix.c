#include <string.h>
#include "alloc.h"
#include "vec.h"
#include "matrix.h"
#include "matrix_impl.h"

__BEGIN_DECLS__

static void MaskVec(value_type* input, value_type* mask, value_type* output, int n) {
	VecPointwiseMult(input, mask, output, n);
} 

/* MatrixCSR API */
MatrixCSR* MatrixCSRCreate(const CSRAttr* attr) {
	MatrixCSR* mat = (MatrixCSR*)CdamMallocHost(sizeof(MatrixCSR));
	mat->attr = attr;
	mat->val = (value_type*)CdamMallocDevice(CSRAttrNNZ(attr) * sizeof(value_type));
	cusparseHandle_t handle;
	cusparseCreate(&handle);

	mat->buffer_size = 0;
	mat->buffer = NULL;
	if(sizeof(value_type) == sizeof(f64)) {
		cusparseCreateCsr(
			&mat->descr,
			CSRAttrNumRow(attr),
			CSRAttrNumCol(attr),
			CSRAttrNNZ(attr),
			CSRAttrRowPtr(attr),
			CSRAttrColInd(attr),
			mat->val,
			CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_BASE_ZERO,
			CUDA_R_64F
		);
	}
	else if(sizeof(value_type) == sizeof(f32)) {
		cusparseCreateCsr(
			&mat->descr,
			CSRAttrNumRow(attr),
			CSRAttrNumCol(attr),
			CSRAttrNNZ(attr),
			CSRAttrRowPtr(attr),
			CSRAttrColInd(attr),
			mat->val,
			CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_BASE_ZERO,
			CUDA_R_32F
		);
	}
	else {
		ASSERT((sizeof(value_type) == sizeof(f64) || sizeof(value_type) == sizeof(f32)) && "Unsupported value_type");
	}
	cusparseDestroy(handle);
	return mat;
}

void MatrixCSRDestroy(Matrix* mat) {
	MatrixCSR* mat_csr = (MatrixCSR*)mat->data;
	cusparseDestroySpMat(mat_csr->descr);
	index_type nnz = CSRAttrNNZ(mat_csr->attr);
	CdamFreeDevice(mat_csr->val, nnz * sizeof(value_type));
	CdamFreeDevice(mat_csr->buffer, mat_csr->buffer_size);
	CdamFreeHost(mat, sizeof(MatrixCSR));
}

/* Zero out the matrix */
static void MatrixCSRZero(Matrix* mat) {
	MatrixCSR* mat_csr = (MatrixCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	const index_type nnz = CSRAttrNNZ(attr);
	cudaMemset(mat_csr->val, 0, nnz * sizeof(value_type));
}


/* Matrix-vector multiplication */
/* y = alpha * A * x + beta * y */
static void MatrixCSRAMVPBY(value_type alpha, Matrix* A, value_type* x, value_type beta, value_type* y) {
	MatrixCSR* mat_csr = (MatrixCSR*)A->data;
	const CSRAttr* attr = mat_csr->attr;
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_idx = CSRAttrColInd(attr);
	const value_type* val = mat_csr->val;
	const index_type num_row = CSRAttrNumRow(attr);
	const index_type num_col = CSRAttrNumCol(attr);

	cusparseHandle_t handle;
	cusparseCreate(&handle);
	cusparseDnVecDescr_t x_desc, y_desc;
	cusparseCreateDnVec(&x_desc, num_col, x, CUDA_R_64F);
	cusparseCreateDnVec(&y_desc, num_row, y, CUDA_R_64F);

	if(mat_csr->buffer_size == 0) {
		size_t buffer_size = 0;
		cusparseSpMV_bufferSize(
			handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			mat_csr->descr,
			x_desc,
			&beta,
			y_desc,
			CUDA_R_64F,
			CUSPARSE_SPMV_ALG_DEFAULT,
			&buffer_size
		);
		mat_csr->buffer_size = (index_type)buffer_size;
		mat_csr->buffer = (void*)CdamMallocDevice(mat_csr->buffer_size);
	}
	cusparseSpMV(
		handle,
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha,
		mat_csr->descr,
		x_desc,
		&beta,
		y_desc,
		CUDA_R_64F,
		CUSPARSE_SPMV_ALG_DEFAULT,
		mat_csr->buffer
	);
	cusparseDestroy(handle);
	cusparseDestroyDnVec(x_desc);
	cusparseDestroyDnVec(y_desc);
}

/* y = left_mask(alpha*A*right_mask(x)+beta*y) */
static void MatrixCSRAMVPBYWithMask(value_type alpha, Matrix* A, value_type* x, value_type beta, value_type* y,
																		value_type* left_mask, value_type* right_mask) {
	MatrixCSR* mat_csr = (MatrixCSR*)A->data;
	const CSRAttr* attr = mat_csr->attr;
	const index_type num_row = CSRAttrNumRow(attr);
	const index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_idx = CSRAttrColInd(attr);
	const value_type* val = mat_csr->val;

	/* Mask the input vector: input = right_mask(x) */
	value_type* input = NULL;
	if(right_mask) {
		input = (value_type*)CdamMallocDevice(num_col * sizeof(value_type));
		MaskVec(x, right_mask, input, num_col);
	}
	else {
		input = (value_type*)x;
	}

	/* Perform matrix-vector multiplication */
	MatrixCSRAMVPBY(alpha, A, input, beta, y);

	/* Mask the output vector: y = left_mask(y) */
	if(left_mask) {
		MaskVec(y, left_mask, y, num_row);
	}

	/* Free the mask vector */
	if(right_mask) {
		CdamFreeDevice(input, num_col * sizeof(value_type));
	}
}

/* y = A * x */
static void MatrixCSRMatVec(Matrix* mat, value_type* x, value_type* y) {
	value_type alpha = 1.0, beta = 0.0;
	MatrixCSRAMVPBY(alpha, mat, x, beta, y);
}

/* y = left_mask(A * right_mask(x)) */
static void MatrixCSRMatVecWithMask(Matrix* mat, value_type* x, value_type* y,
														 value_type* left_mask, value_type* right_mask) {
	value_type alpha = 1.0, beta = 0.0;
	MatrixCSRAMVPBYWithMask(alpha, mat, x, beta, y, left_mask, right_mask);
}

/* diag = diag(A) */
static void MatrixCSRGetDiag(Matrix* mat, value_type* diag) {
	MatrixCSR* mat_csr = (MatrixCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	const index_type num_row = CSRAttrNumRow(attr);
	const index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_idx = CSRAttrColInd(attr);
	const value_type* val = mat_csr->val;

	MatrixCSRGetDiagGPU(val, row_ptr, col_idx, diag, num_row);
}

static void MatrixCSRAddElementLHS(Matrix* mat, index_type nshl, index_type bs,
																	 index_type batch_size, const index_type* batch_index_ptr,
																	 const index_type* ien,
																	 const value_type* val, int lda) {
	MatrixCSR* mat_csr = (MatrixCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	index_type num_row = CSRAttrNumRow(attr);
	index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_ind = CSRAttrColInd(attr);

	MatrixCSRAddElementLHSGPU(mat_csr->val, nshl, bs,
														num_row, row_ptr, num_col, col_ind,
														batch_size, batch_index_ptr, ien,
														val, lda);
}


/* MatrixNested API */
MatrixNested* MatrixNestedCreate(index_type n_offset, const index_type* offset) {
	size_t mem_size = sizeof(MatrixNested) + sizeof(Matrix*) * n_offset * n_offset;
	MatrixNested* mat = (MatrixNested*)CdamMallocHost(mem_size);
	mat->n_offset = n_offset;
	mat->offset = offset;
	memset(mat->mat, 0, sizeof(Matrix*) * n_offset * n_offset);
	return mat;
}

void MatrixNestedDestroy(Matrix* mat) {
	MatrixNested* mat_nested = (MatrixNested*)mat->data;
	index_type n = mat_nested->n_offset;
	for(index_type i = 0; i < n * n; i++) {
		if(mat_nested->mat[i]) {
			MatrixDestroy(mat_nested->mat[i]);
		}
	}
	CdamFreeHost(mat_nested, sizeof(MatrixNested) + sizeof(Matrix*) * n * n);
	CdamFreeHost(mat, sizeof(Matrix));
}

static void MatrixNestedZero(Matrix* mat) {
	MatrixNested* mat_nested = (MatrixNested*)mat->data;
	index_type n_offset = mat_nested->n_offset;
	const index_type* offset = mat_nested->offset;
	Matrix** mat_list = mat_nested->mat;

	/* Zero out the matrix */
	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			MatrixZero(mat_list[i * n_offset + j]);
		}
	}
}

static void MatrixNestedAMVPBY(value_type alpha, Matrix* A, value_type* x, value_type beta, value_type* y) {
	MatrixNested* mat_nested = (MatrixNested*)A->data;
	index_type n_offset = mat_nested->n_offset;
	const index_type* offset = mat_nested->offset;
	Matrix** mat_list = mat_nested->mat;

	/* Perform matrix-vector multiplication */
	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			MatrixAMVPBY(alpha, mat_list[i * n_offset + j], x + offset[j], beta, y + offset[i]);
		}
	}
}

static void MatrixNestedAMVPBYWithMask(value_type alpha, Matrix* A, value_type* x, value_type beta, value_type* y,
																			 value_type* left_mask, value_type* right_mask) {
	MatrixNested* mat_nested = (MatrixNested*)A->data;
	index_type n_offset = mat_nested->n_offset;
	const index_type* offset = mat_nested->offset;
	Matrix** mat_list = mat_nested->mat;
	value_type* lm = left_mask, *rm = right_mask;

	/* Perform matrix-vector multiplication */
	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			MatrixAMVPBYWithMask(alpha, mat_list[i * n_offset + j], x + offset[j], beta, y + offset[i],
													 (lm ? lm + offset[i] : NULL), (rm ? rm + offset[j] : NULL));
		}
	}
}

static void MatrixNestedMatVec(Matrix* mat, value_type* x, value_type* y) {
	value_type alpha = 1.0, beta = 0.0;
	MatrixNestedAMVPBY(alpha, mat, x, beta, y);
}

static void MatrixNestedMatVecWithMask(Matrix* mat, value_type* x, value_type* y,
																			 value_type* left_mask, value_type* right_mask) {
	value_type alpha = 1.0, beta = 0.0;
	MatrixNestedAMVPBYWithMask(alpha, mat, x, beta, y, left_mask, right_mask);
}

static void MatrixNestedGetDiag(Matrix* mat, value_type* diag) {
	MatrixNested* mat_nested = (MatrixNested*)mat->data;
	index_type n_offset = mat_nested->n_offset;
	const index_type* offset = mat_nested->offset;
	Matrix** mat_list = mat_nested->mat;
	
	/* Set to zero */
	cudaMemset(diag, 0, sizeof(value_type) * offset[n_offset]);
	/* Get diagonal elements */
	for(index_type i = 0; i < n_offset; i++) {
		if(mat_list[i * n_offset + i] == NULL) {
			continue;
		}
		MatrixGetDiag(mat_list[i * n_offset + i], diag + offset[i]);
	}
}

static void MatrixNestedAddElementLHS(Matrix* mat, index_type nshl, index_type bs,
																			index_type batch_size, const index_type* batch_index_ptr, const index_type* ien,
																			const value_type* val, int lda) {
	MatrixNested* mat_nested = (MatrixNested*)mat->data;
	index_type n_offset = mat_nested->n_offset;
	const index_type* offset = mat_nested->offset;
	Matrix** mat_list = mat_nested->mat;

	/* Add element to the matrix */
	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			MatrixAddElementLHS(mat_list[i * n_offset + j], nshl, bs,
													batch_size, batch_index_ptr, ien,
													val + offset[i] * lda + offset[j] * nshl, lda);
		}
	}
}

/* Matrix API */
Matrix* MatrixCreateTypeCSR(const CSRAttr* attr) {
	Matrix *mat = (Matrix*)CdamMallocHost(sizeof(Matrix));
	mat->size[0] = CSRAttrNumRow(attr);
	mat->size[1] = CSRAttrNumCol(attr);

	/* Set up the matrix type */
	mat->type = MAT_TYPE_CSR;

	/* Set up the matrix data */	
	mat->data = MatrixCSRCreate(attr);

	/* Set up the matrix operation */
	// mat->op = &mat->_op_private;
	mat->op->zero = MatrixCSRZero;
	mat->op->amvpby = MatrixCSRAMVPBY;
	mat->op->amvpby_mask = MatrixCSRAMVPBYWithMask;
	mat->op->matvec = MatrixCSRMatVec;
	mat->op->matvec_mask = MatrixCSRMatVecWithMask;
	mat->op->get_diag = MatrixCSRGetDiag;
	mat->op->add_element_lhs = MatrixCSRAddElementLHS;
	mat->op->destroy = MatrixCSRDestroy;
	return mat;
}



Matrix* MatrixCreateTypeNested(index_type n_offset, const index_type* offset) {
	Matrix* mat = (Matrix*)CdamMallocHost(sizeof(Matrix));
	mat->size[0] = offset[n_offset];
	mat->size[1] = offset[n_offset];

	/* Set up the matrix type */
	mat->type = MAT_TYPE_NESTED;

	/* Set up the matrix data */
	mat->data = MatrixNestedCreate(n_offset, offset);

	/* Set up the matrix operation */
	// mat->op = &mat->_op_private;
	mat->op->zero = MatrixNestedZero;	
	mat->op->amvpby = MatrixNestedAMVPBY;
	mat->op->amvpby_mask = MatrixNestedAMVPBYWithMask;
	mat->op->matvec = MatrixNestedMatVec;
	mat->op->matvec_mask = MatrixNestedMatVecWithMask;
	mat->op->get_diag = MatrixNestedGetDiag;
	mat->op->add_element_lhs = MatrixNestedAddElementLHS;
	mat->op->destroy = MatrixNestedDestroy;
	return mat;
}

void MatrixDestroy(Matrix* mat) {
	if(mat == NULL) {
		return;
	}
	if(mat->op->destroy) {
		mat->op->destroy(mat);
	}
}

void MatrixZero(Matrix* mat) {
	ASSERT(mat && "Matrix is NULL");
	if(mat->op->zero) {
		mat->op->zero(mat);
	}
}


void MatrixAMVPBY(value_type alpha, Matrix* A, value_type* x, value_type beta, value_type* y) {
	ASSERT(A && "Matrix is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	if(A->op->amvpby) {
		A->op->amvpby(alpha, A, x, beta, y);
	}
}

void MatrixAMVPBYWithMask(value_type alpha, Matrix* A, value_type* x, value_type beta, value_type* y,
													value_type* left_mask, value_type* right_mask) {
	ASSERT(A && "Matrix is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	if(A->op->amvpby_mask) {
		A->op->amvpby_mask(alpha, A, x, beta, y, left_mask, right_mask);
	}
}


void MatrixMatVec(Matrix* mat, value_type* x, value_type* y) {
	ASSERT(mat && "Matrix is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	if(mat->op->matvec) {
		mat->op->matvec(mat, x, y);
	}
}

void MatrixMatVecWithMask(Matrix* mat, value_type* x, value_type* y, value_type* left_mask, value_type* right_mask) {
	ASSERT(mat && "Matrix is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	if(mat->op->matvec_mask) {
		mat->op->matvec_mask(mat, x, y, left_mask, right_mask);
	}
}

void MatrixGetDiag(Matrix* mat, value_type* diag) {
	ASSERT(mat && "Matrix is NULL");
	ASSERT(diag && "diag is NULL");
	if(mat->op->get_diag) {
		mat->op->get_diag(mat, diag);
	}
}

void MatrixAddElementLHS(Matrix* mat, index_type nshl, index_type bs,
												 index_type batch_size, const index_type* batch_ptr, const index_type* ien,
												 const value_type* val, int lda) {
	ASSERT(mat && "Matrix is NULL");
	if(mat->op->add_element_lhs) {
		mat->op->add_element_lhs(mat, nshl, bs,
														 batch_size, batch_ptr, ien,
														 val, lda);
	}
}

__END_DECLS__
