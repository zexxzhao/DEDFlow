#include <string.h>
#include <cublas_v2.h>
#include "alloc.h"
#include "vec.h"
#include "matrix.h"
#include "matrix_impl.h"

__BEGIN_DECLS__

static void MaskVec(value_type* input, value_type* mask, value_type* output, int n) {
	VecPointwiseMult(input, mask, output, n);
} 

/****************************************************
 * MatrixCSR Operation
 ****************************************************/
MatrixCSR* MatrixCSRCreate(const CSRAttr* attr, void* handle) {
	MatrixCSR* mat = (MatrixCSR*)CdamMallocHost(SIZE_OF(MatrixCSR));
	mat->attr = attr;
	mat->val = (value_type*)CdamMallocDevice(CSRAttrNNZ(attr) * SIZE_OF(value_type));
	CUGUARD(cudaGetLastError());
	mat->handle = *(cusparseHandle_t*)handle;
	cudaMemset(mat->val, 0, CSRAttrNNZ(attr) * SIZE_OF(value_type));
	CUGUARD(cudaGetLastError());

	mat->buffer_size = 0;
	mat->buffer = NULL;
	if(SIZE_OF(value_type) == SIZE_OF(f64)) {
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
	else if(SIZE_OF(value_type) == SIZE_OF(f32)) {
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
		ASSERT((SIZE_OF(value_type) == SIZE_OF(f64) || SIZE_OF(value_type) == SIZE_OF(f32)) && "Unsupported value_type");
	}
	return mat;
}

void MatrixCSRDestroy(Matrix* mat) {
	MatrixCSR* mat_csr = (MatrixCSR*)mat->data;
	cusparseDestroySpMat(mat_csr->descr);
	index_type nnz = CSRAttrNNZ(mat_csr->attr);
	CdamFreeDevice(mat_csr->val, nnz * SIZE_OF(value_type));
	CdamFreeDevice(mat_csr->buffer, mat_csr->buffer_size);
	CdamFreeHost(mat, SIZE_OF(MatrixCSR));
}

static void MatrixCSRSetup(Matrix* mat) {
}

/* Zero out the matrix */
static void MatrixCSRZero(Matrix* mat) {
	MatrixCSR* mat_csr = (MatrixCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	const index_type nnz = CSRAttrNNZ(attr);
	cudaMemsetAsync(mat_csr->val, 0, nnz * SIZE_OF(value_type), 0);
}

/* Zero the rows of the matrix */
static void MatrixCSRZeroRow(Matrix* mat, index_type n, const index_type* row, index_type shift, value_type diag) {
	MatrixCSR* mat_csr = (MatrixCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	const index_type num_row = CSRAttrNumRow(attr);
	const index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_idx = CSRAttrColInd(attr);
	value_type* val = mat_csr->val;

	MatrixCSRZeroRowGPU(val,
											num_row, num_col, row_ptr, col_idx,
											n, row, shift, diag);
}


/* Matrix-vector multiplication */
/* y = alpha * A * x + beta * y */
static void MatrixCSRAMVPBY(Matrix* A, value_type alpha, value_type* x, value_type beta, value_type* y) {
	MatrixCSR* mat_csr = (MatrixCSR*)A->data;
	const CSRAttr* attr = mat_csr->attr;
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_idx = CSRAttrColInd(attr);
	const value_type* val = mat_csr->val;
	const index_type num_row = CSRAttrNumRow(attr);
	const index_type num_col = CSRAttrNumCol(attr);

	cusparseDnVecDescr_t x_desc, y_desc;
	cusparseCreateDnVec(&x_desc, num_col, x, CUDA_R_64F);
	cusparseCreateDnVec(&y_desc, num_row, y, CUDA_R_64F);

	cusparseSpMVAlg_t alg =  CUSPARSE_SPMV_CSR_ALG2;

	if(mat_csr->buffer_size == 0) {
		size_t buffer_size = 0;
		cusparseSpMV_bufferSize(
			mat_csr->handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			mat_csr->descr,
			x_desc,
			&beta,
			y_desc,
			CUDA_R_64F,
			alg,
			&buffer_size
		);
		mat_csr->buffer_size = (index_type)buffer_size;
		mat_csr->buffer = (void*)CdamMallocDevice(mat_csr->buffer_size);
		/* If cuda version is 12.4 or higher, use cusparseSpMV_preprocess */ 
#if CUDART_VERSION >= 12040
		cusparseSpMV_preprocess(
			mat_csr->handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha,
			mat_csr->descr,
			x_desc,
			&beta,
			y_desc,
			CUDA_R_64F,
			alg,
			mat_csr->buffer
		);
#endif
	}
	// cusparseSetStream(mat_csr->handle, A->stream_ref);
	
	cusparseSpMV(
		mat_csr->handle,
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha,
		mat_csr->descr,
		x_desc,
		&beta,
		y_desc,
		CUDA_R_64F,
		alg,
		mat_csr->buffer
	);
	cusparseDestroyDnVec(x_desc);
	cusparseDestroyDnVec(y_desc);
}

/* y = left_mask(alpha*A*right_mask(x)+beta*y) */
static void MatrixCSRAMVPBYWithMask(Matrix* A, value_type alpha, value_type* x, value_type beta, value_type* y,
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
		input = (value_type*)CdamMallocDevice(num_col * SIZE_OF(value_type));
		MaskVec(x, right_mask, input, num_col);
	}
	else {
		input = (value_type*)x;
	}

	/* Perform matrix-vector multiplication */
	MatrixCSRAMVPBY(A, alpha, input, beta, y);

	/* Mask the output vector: y = left_mask(y) */
	if(left_mask) {
		MaskVec(y, left_mask, y, num_row);
	}

	/* Free the mask vector */
	if(right_mask) {
		CdamFreeDevice(input, num_col * SIZE_OF(value_type));
	}
}

/* y = A * x */
static void MatrixCSRMatVec(Matrix* mat, value_type* x, value_type* y) {
	value_type alpha = 1.0, beta = 0.0;
	MatrixCSRAMVPBY(mat, alpha, x, beta, y);
}

/* y = left_mask(A * right_mask(x)) */
static void MatrixCSRMatVecWithMask(Matrix* mat, value_type* x, value_type* y,
														 value_type* left_mask, value_type* right_mask) {
	value_type alpha = 1.0, beta = 0.0;
	MatrixCSRAMVPBYWithMask(mat, alpha, x, beta, y, left_mask, right_mask);
}

/* diag = diag(A) */
static void MatrixCSRGetDiag(Matrix* mat, value_type* diag, index_type bs) {
	MatrixCSR* mat_csr = (MatrixCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	const index_type num_row = CSRAttrNumRow(attr);
	const index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_idx = CSRAttrColInd(attr);
	const value_type* val = mat_csr->val;
	ASSERT(num_row == num_col && "Matrix is not square");

	if(bs == 1) {
		MatrixCSRGetDiagGPU(val, row_ptr, col_idx, diag, num_row);
	}
	else if(bs > 1) {
		if(attr->num_row != attr->parent->num_row * bs) {
			ASSERT(0 && "Block size is not compatible with the matrix size");
		}	
		attr = attr->parent;
		MatrixGetDiagBlockGPU(mat_csr->val, bs,
													attr->num_row, attr->num_col, attr->row_ptr, attr->col_ind,
													diag, bs, bs * bs);
													
	}
	else {
		ASSERT(0 && "Block size should be greater than 0");
	}
}

static void MatrixCSRSetValuesCOO(Matrix* mat, value_type alpha,
																	index_type n, const index_type* row, const index_type* col,
																	const value_type* val, value_type beta) {
	MatrixCSR* mat_csr = (MatrixCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	index_type num_row = CSRAttrNumRow(attr);
	index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_ind = CSRAttrColInd(attr);

	MatrixCSRSetValuesCOOGPU(mat_csr->val, alpha,
													 num_row, num_col, row_ptr, col_ind,
													 n, row, col, val, beta);
}

static void MatrixCSRAddElemValueBatched(Matrix* matrix, index_type nshl,
																				 index_type batch_size, const index_type* batch_ptr, const index_type* ien,
																				 const value_type* val, const index_type* mask) {
	MatrixCSR* mat_csr = (MatrixCSR*)matrix->data;
	const CSRAttr* attr = mat_csr->attr;
	index_type num_row = CSRAttrNumRow(attr);
	index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_ind = CSRAttrColInd(attr);

	MatrixCSRAddElemValueBatchedGPU(mat_csr->val, 1.0,
																	batch_size, batch_ptr, ien, nshl,
																	num_row, num_col, row_ptr, col_ind,
																	val, 1.0, mask);
}

static void MatrixCSRAddElemValueBlockedBatched(Matrix* matrix, index_type nshl,
																								index_type batch_size, const index_type* batch_index_ptr, const index_type* ien,
																								index_type block_row, index_type block_col,
																								const value_type* val, int lda, int stride, const index_type* mask) {
	MatrixCSR* mat_csr = (MatrixCSR*)matrix->data;
	const CSRAttr* attr = mat_csr->attr;
	index_type num_row = CSRAttrNumRow(attr);
	index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_ind = CSRAttrColInd(attr);

	MatrixCSRAddElemValueBlockedBatchedGPU(mat_csr->val, 1.0, 
																				 batch_size, batch_index_ptr, ien, nshl,
																				 num_row, num_col, row_ptr, col_ind,
																				 block_row, block_col,
																				 val, lda, stride, 1.0, mask);
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

static void MatrixCSRAddValueBatched(Matrix* mat,
																		 index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
																		 const value_type* A) {
	MatrixCSR* mat_csr = (MatrixCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	index_type csr_num_row = CSRAttrNumRow(attr);
	index_type csr_num_col = CSRAttrNumCol(attr);
	const index_type* csr_row_ptr = CSRAttrRowPtr(attr);
	const index_type* csr_col_ind = CSRAttrColInd(attr);

	MatrixCSRSetValueBatchedGPU(mat_csr->val, 1.0,
															csr_num_row, csr_num_col, csr_row_ptr, csr_col_ind, /* CSR */
															batch_size, batch_row_ind, batch_col_ind, /* Batched */
															A, 1.0);
}

static void MatrixCSRAddValueBlockedBatched(Matrix* mat,
																						index_type batch_size,
																						const index_type* batch_row_ind, const index_type* batch_col_ind,
																						index_type block_row, index_type block_col,
																						const value_type* A, int lda, int stride) {
	MatrixCSR* mat_csr = (MatrixCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	index_type csr_num_row = CSRAttrNumRow(attr);
	index_type csr_num_col = CSRAttrNumCol(attr);
	const index_type* csr_row_ptr = CSRAttrRowPtr(attr);
	const index_type* csr_col_ind = CSRAttrColInd(attr);

	MatrixCSRSetValueBlockedBatchedGPU(mat_csr->val, 1.0,
																			csr_num_row, csr_num_col, csr_row_ptr, csr_col_ind, /* CSR */
																			batch_size, batch_row_ind, batch_col_ind, /* Batched */
																			block_row, block_col, /* Block */
																			A, 1.0, lda, stride);
}



/****************************************************
 * MatrixFS Operation
 ****************************************************/

MatrixFS* MatrixFSCreate(index_type n_offset, const index_type* offset, void* handle) {
	MatrixFS* mat = (MatrixFS*)CdamMallocHost(SIZE_OF(MatrixFS));
	memset(mat, 0, SIZE_OF(MatrixFS));

	mat->n_offset = n_offset;

	mat->offset = (index_type*)CdamMallocHost(SIZE_OF(index_type) * (n_offset + 1));
	memcpy(mat->offset, offset, SIZE_OF(index_type) * (n_offset + 1));

	mat->d_offset = (index_type*)CdamMallocDevice((n_offset + 1) * SIZE_OF(index_type));
	cudaMemcpy(mat->d_offset, offset, (n_offset + 1) * SIZE_OF(index_type), cudaMemcpyHostToDevice);

	mat->d_matval = (value_type**)CdamMallocDevice(SIZE_OF(value_type*) * n_offset * n_offset);
	cudaMemset(mat->d_matval, 0, SIZE_OF(value_type*) * n_offset * n_offset);

	mat->mat = (Matrix**)CdamMallocHost(SIZE_OF(Matrix*) * n_offset * n_offset);
	memset(mat->mat, 0, SIZE_OF(Matrix*) * n_offset * n_offset);

	mat->handle = *(cublasHandle_t*)handle;
	mat->stream = (cudaStream_t*)CdamMallocHost(SIZE_OF(cudaStream_t) * n_offset);
	for(index_type i = 0; i < n_offset; i++) {
		cudaStreamCreate(mat->stream + i);
	}
	
	return mat;
}

void MatrixFSDestroy(Matrix* mat) {
	MatrixFS* mat_fs = (MatrixFS*)mat->data;
	index_type n = mat_fs->n_offset;
	for(index_type i = 0; i < n * n; i++) {
		if(mat_fs->mat[i]) {
			MatrixDestroy(mat_fs->mat[i]);
		}
	}
	CdamFreeHost(mat_fs->offset, (n + 1) * SIZE_OF(index_type));
	CdamFreeDevice(mat_fs->d_offset, (n + 1) * SIZE_OF(index_type));
	CdamFreeDevice(mat_fs->d_matval, n * n * SIZE_OF(value_type*));
	CdamFreeHost(mat_fs->mat, n * n * SIZE_OF(Matrix*));
	CdamFreeHost(mat_fs, SIZE_OF(MatrixFS));
	CdamFreeHost(mat, SIZE_OF(Matrix));

	for(index_type i = 0; i < n; i++) {
		cudaStreamDestroy(mat_fs->stream[i]);
	}
	CdamFreeHost(mat_fs->stream, SIZE_OF(cudaStream_t) * n);
}

static void MatrixFSSetup(Matrix* mat) {
	MatrixFS* mat_fs = (MatrixFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	index_type num_row = mat_fs->spy1x1->num_row;
	index_type num_col = mat_fs->spy1x1->num_col;

	Matrix *submat = NULL;

	/* Set the matrix size */
	mat->size[0] = offset[n_offset] * num_row;
	mat->size[1] = offset[n_offset] * num_col;

	/* Set up the submatrices */
	for(index_type i = 0; i < n_offset * n_offset; i++) {
		if(mat_fs->mat[i]) {
			MatrixSetup(mat_fs->mat[i]);
		}
	}

	/* Gather the matrix values */
	value_type** matval = (value_type**)CdamMallocHost(SIZE_OF(value_type*) * n_offset * n_offset);
	memset(matval, 0, SIZE_OF(value_type*) * n_offset * n_offset);
	for(index_type i = 0; i < n_offset * n_offset; i++) {
		submat = mat_fs->mat[i];
		if(submat == NULL) {
			continue;
		}
		matval[i] = ((MatrixCSR*)submat->data)->val;
	}
	cudaMemcpy(mat_fs->d_matval, matval, SIZE_OF(value_type*) * n_offset * n_offset, cudaMemcpyHostToDevice);
	CdamFreeHost(matval, SIZE_OF(value_type*) * n_offset * n_offset);
}

static void MatrixFSZero(Matrix* mat) {
	MatrixFS* mat_fs = (MatrixFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	Matrix** mat_list = mat_fs->mat;

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

static void MatrixFSZeroRow(Matrix* mat, index_type n, const index_type* row, index_type shift, value_type diag) {
	MatrixFS* mat_fs = (MatrixFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	index_type num_row = mat_fs->spy1x1->num_row;
	Matrix** mat_list = mat_fs->mat;

	// ASSERT(0 && "Not implemented");

	/* Zero out the matrix */
	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			MatrixZeroRow(mat_list[i * n_offset + j], n - offset[i] * num_row, row + offset[i] * num_row,
										-num_row * offset[i],
										i == j ? diag: 0.0 );
		}
	}
}

static void MatrixFSAMVPBY(Matrix* A, value_type alpha, value_type* x, value_type beta, value_type* y) {
	MatrixFS* mat_fs = (MatrixFS*)A->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	index_type num_row = mat_fs->spy1x1->num_row;
	index_type num_col = mat_fs->spy1x1->num_col;
	Matrix** mat_list = mat_fs->mat;

	cublasDscal(mat_fs->handle, n_offset * num_row, &beta, y, 1);


	/* Perform matrix-vector multiplication */
	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			mat_list[i * n_offset + j]->stream_ref = NULL; //mat_fs->stream[i];
			cudaStreamSynchronize(0);
			MatrixAMVPBY(mat_list[i * n_offset + j], alpha, x + offset[j] * num_col, 1.0, y + offset[i] * num_row);
			cudaStreamSynchronize(0);
			mat_list[i * n_offset + j]->stream_ref = NULL;
		}
	}

}

static void MatrixFSAMVPBYWithMask(Matrix* A, value_type alpha, value_type* x, value_type beta, value_type* y,
																	 value_type* left_mask, value_type* right_mask) {
	MatrixFS* mat_fs = (MatrixFS*)A->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	index_type num_row = mat_fs->spy1x1->num_row;
	index_type num_col = mat_fs->spy1x1->num_col;
	Matrix** mat_list = mat_fs->mat;
	value_type* lm = left_mask, *rm = right_mask;

	/* Perform matrix-vector multiplication */
	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			MatrixAMVPBYWithMask(mat_list[i * n_offset + j], alpha, x + offset[j] * num_col, beta, y + offset[i] * num_row,
													 (lm ? lm + offset[i] * num_col : NULL), (rm ? rm + offset[j] * num_row : NULL));
		}
	}
}

static void MatrixFSMatVec(Matrix* mat, value_type* x, value_type* y) {
	value_type alpha = 1.0, beta = 0.0;
	MatrixFSAMVPBY(mat, alpha, x, beta, y);
}

static void MatrixFSMatVecWithMask(Matrix* mat, value_type* x, value_type* y,
																			 value_type* left_mask, value_type* right_mask) {
	value_type alpha = 1.0, beta = 0.0;
	MatrixFSAMVPBYWithMask(mat, alpha, x, beta, y, left_mask, right_mask);
}

static void MatrixFSGetDiag(Matrix* mat, value_type* diag, index_type bs) {
	MatrixFS* mat_fs = (MatrixFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	index_type num_row = mat_fs->spy1x1->num_row;
	Matrix** mat_list = mat_fs->mat;
	
	/* Set to zero */
	cudaMemset(diag, 0, SIZE_OF(value_type) * offset[n_offset] * num_row);
	/* Get diagonal elements */
	for(index_type i = 0; i < n_offset; i++) {
		if(mat_list[i * n_offset + i] == NULL) {
			continue;
		}
		MatrixGetDiag(mat_list[i * n_offset + i], diag + offset[i] * num_row, 1);
	}


	// CdamFreeDevice(nzind, batch_size * SIZE_OF(index_type));
}

static void MatrixFSSetValuesCOO(Matrix* mat, value_type alpha,
																		 index_type n, const index_type* row, const index_type* col,
																		 const value_type* val, value_type beta) {
	MatrixFS* mat_fs = (MatrixFS*)mat->data;
	ASSERT(0 && "Not implemented");
}
static void MatrixFSSetValuesInd(Matrix* mat, value_type alpha,
																		 index_type n, const index_type* ind,
																		 const value_type* val, value_type beta) {
	MatrixFS* mat_fs = (MatrixFS*)mat->data;
	ASSERT(0 && "Not implemented");
}


static void MatrixFSAddElemValueBatched(Matrix* mat, index_type nshl,
																						index_type batch_size, const index_type* batch_index_ptr,
																						const index_type* ien, const value_type* val, const index_type* mask) {
	ASSERT(0 && "Not supported yet");

}

static void MatrixFSAddElemValueBlockedBatched(Matrix* mat, index_type nshl,
																							 index_type batch_size, const index_type* batch_index_ptr,
																							 const index_type* ien,
																							 index_type block_row, index_type block_col,
																							 const value_type* val, int lda, int stride, const index_type* mask) {
	MatrixFS* mat_fs = (MatrixFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->d_offset;
	
	value_type** matval = mat_fs->d_matval;
	const CSRAttr* spy1x1 = mat_fs->spy1x1;

	SetBlockValueToSubmatGPU(matval, 1.0,
													 n_offset, offset, nshl,
													 batch_size, batch_index_ptr, ien,
													 spy1x1->num_row, spy1x1->num_col,
													 spy1x1->row_ptr, spy1x1->col_ind,
													 val, lda, stride, 1.0, mask);
}
																						

static void MatrixFSAddElementLHS(Matrix* mat, index_type nshl, index_type bs,
																	index_type batch_size, const index_type* batch_index_ptr, const index_type* ien,
																			const value_type* val, int lda) {
	MatrixFS* mat_fs = (MatrixFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	Matrix** mat_list = mat_fs->mat;
	ASSERT(0 && "Not implemented yet");

	// MatrixFSAddElementLHSGPU(mat_list, n_offset, offset, nshl, bs,
	// 														 batch_size, batch_index_ptr, ien,
	// 														 val, lda);
}

static void MatrixFSAddValueBatched(Matrix* mat,
																		index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
																		const value_type* A, int lda, int stride) {

	ASSERT(0 && "Not implemented yet");
}

static void MatrixFSAddValueBlockedBatched(Matrix* mat,
																					 index_type batch_size,
																					 const index_type* batch_row_ind, const index_type* batch_col_ind,
																					 index_type block_row, index_type block_col,
																					 const value_type* A, int lda, int stride) {
	MatrixFS* mat_fs = (MatrixFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	Matrix** mat_list = mat_fs->mat;

	const CSRAttr* attr = mat_fs->spy1x1;
	ASSERT(attr && "CSRAttr is NULL");

	// index_type* nzind = (index_type*)CdamMallocDevice(batch_size * SIZE_OF(index_type));
	

	// CSRAttrGetNZIndBatched(attr, batch_size, batch_row_ind, batch_col_ind, nzind);

	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			MatrixAddValueBlockedBatched(mat_list[i * n_offset + j],
																	 batch_size, batch_row_ind, batch_col_ind,
																	 offset[i + 1] - offset[i], offset[j + 1] - offset[j],
																	 A + offset[i] * lda + offset[j],
																	 lda, stride);
		}
	}


	// CdamFreeDevice(nzind, batch_size * SIZE_OF(index_type));
}

/****************************************************
 * General Matrix Operation
 ****************************************************/
Matrix* MatrixCreateTypeCSR(const CSRAttr* attr, void* handle) {
	Matrix *mat = (Matrix*)CdamMallocHost(SIZE_OF(Matrix));
	mat->size[0] = CSRAttrNumRow(attr);
	mat->size[1] = CSRAttrNumCol(attr);

	/* Set up the matrix type */
	mat->type = MAT_TYPE_CSR;

	/* Set up the matrix data */	
	mat->data = MatrixCSRCreate(attr, handle);

	/* Set up the matrix operation */
	mat->op->setup = MatrixCSRSetup;

	mat->op->zero = MatrixCSRZero;
	mat->op->zero_row = MatrixCSRZeroRow;

	mat->op->amvpby = MatrixCSRAMVPBY;
	mat->op->amvpby_mask = MatrixCSRAMVPBYWithMask;
	mat->op->matvec = MatrixCSRMatVec;
	mat->op->matvec_mask = MatrixCSRMatVecWithMask;

	mat->op->get_diag = MatrixCSRGetDiag;

	mat->op->set_values_coo = MatrixCSRSetValuesCOO;
	mat->op->set_values_ind = NULL; /* MatrixCSRSetValuesInd;*/

	mat->op->add_elem_value_batched = MatrixCSRAddElemValueBatched;
	mat->op->add_elem_value_blocked_batched = MatrixCSRAddElemValueBlockedBatched;

	mat->op->add_value_batched = NULL; /* MatrixCSRAddValueBatched; */
	mat->op->add_value_blocked_batched = NULL; /* MatrixCSRAddValueBlockedBatched; */

	mat->op->destroy = MatrixCSRDestroy;
	return mat;
}



Matrix* MatrixCreateTypeFS(index_type n_offset, const index_type* offset, void* handle) {
	Matrix* mat = (Matrix*)CdamMallocHost(SIZE_OF(Matrix));
	mat->size[0] = offset[n_offset];
	mat->size[1] = offset[n_offset];

	/* Set up the matrix type */
	mat->type = MAT_TYPE_FS;

	/* Set up the matrix data */
	mat->data = MatrixFSCreate(n_offset, offset, handle);

	/* Set up the matrix operation */
	mat->op->setup = MatrixFSSetup;

	mat->op->zero = MatrixFSZero;	
	mat->op->zero_row = MatrixFSZeroRow;

	mat->op->amvpby = MatrixFSAMVPBY;
	mat->op->amvpby_mask = MatrixFSAMVPBYWithMask;
	mat->op->matvec = MatrixFSMatVec;
	mat->op->matvec_mask = MatrixFSMatVecWithMask;

	mat->op->get_diag = MatrixFSGetDiag;

	mat->op->set_values_coo = NULL; /* MatrixFSSetValuesCOO; */
	mat->op->set_values_ind = NULL; /* MatrixFSSetValuesInd; */

	mat->op->add_elem_value_batched = MatrixFSAddElemValueBatched;
	mat->op->add_elem_value_blocked_batched = MatrixFSAddElemValueBlockedBatched;

	mat->op->add_value_batched = NULL; /* MatrixFSAddValueBatched; */
	mat->op->add_value_blocked_batched = MatrixFSAddValueBlockedBatched;

	mat->op->destroy = MatrixFSDestroy;
	return mat;
}

#define MATRIX_CALL(mat, func, ...) \
	do { \
		if(mat->op->func) { \
			mat->op->func(mat, ##__VA_ARGS__); \
		} \
		else { \
			fprintf(stderr, "Matrix operation %s is not implemented for type: %d\n", #func, mat->type); \
		} \
	} while(0)

void MatrixDestroy(Matrix* mat) {
	if(mat == NULL) {
		return;
	}
	MATRIX_CALL(mat, destroy);
}

void MatrixSetup(Matrix* mat) {
	ASSERT(mat && "Matrix is NULL");
	MATRIX_CALL(mat, setup);
}

void MatrixZero(Matrix* mat) {
	ASSERT(mat && "Matrix is NULL");
	MATRIX_CALL(mat, zero);
}

void MatrixZeroRow(Matrix* mat, index_type n, const index_type* row, index_type shift, value_type diag) {
	ASSERT(mat && "Matrix is NULL");
	MATRIX_CALL(mat, zero_row, n, row, shift, diag);
}

void MatrixAMVPBY(Matrix* A, value_type alpha, value_type* x, value_type beta, value_type* y) {
	ASSERT(A && "Matrix is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	MATRIX_CALL(A, amvpby, alpha, x, beta, y);
}

void MatrixAMVPBYWithMask(Matrix* A, value_type alpha, value_type* x, value_type beta, value_type* y,
													value_type* left_mask, value_type* right_mask) {
	ASSERT(A && "Matrix is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	MATRIX_CALL(A, amvpby_mask, alpha, x, beta, y, left_mask, right_mask);
}


void MatrixMatVec(Matrix* mat, value_type* x, value_type* y) {
	ASSERT(mat && "Matrix is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	MATRIX_CALL(mat, matvec, x, y);
}

void MatrixMatVecWithMask(Matrix* mat, value_type* x, value_type* y, value_type* left_mask, value_type* right_mask) {
	ASSERT(mat && "Matrix is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	MATRIX_CALL(mat, matvec_mask, x, y, left_mask, right_mask);
}

void MatrixGetDiag(Matrix* mat, value_type* diag, index_type bs) {
	ASSERT(mat && "Matrix is NULL");
	ASSERT(diag && "diag is NULL");
	MATRIX_CALL(mat, get_diag, diag, bs);
}

void MatrixSetValuesCOO(Matrix* mat, value_type alpha,
											  index_type n, const index_type* row, const index_type* col,
											  const value_type* val, value_type beta) {
	ASSERT(mat && "Matrix is NULL");
	MATRIX_CALL(mat, set_values_coo, alpha, n, row, col, val, beta);
}

void MatrixSetValuesInd(Matrix* mat, value_type alpha,
												index_type n, const index_type* ind,
												const value_type* val, value_type beta) {
	ASSERT(mat && "Matrix is NULL");
	MATRIX_CALL(mat, set_values_ind, alpha, n, ind, val, beta);
}

void MatrixAddElemValueBatched(Matrix* mat, index_type nshl,
															 index_type batch_size, const index_type* batch_index_ptr,
															 const index_type* ien, const value_type* val, const index_type* mask) {
	ASSERT(mat && "Matrix is NULL");
	MATRIX_CALL(mat, add_elem_value_batched, nshl, batch_size, batch_index_ptr, ien, val, mask);
}

void MatrixAddElemValueBlockedBatched(Matrix* mat, index_type nshl,
																			index_type batch_size, const index_type* batch_index_ptr,
																			const index_type* ien,
																			index_type block_row, index_type block_col,
																			const value_type* val, int lda, int stride, const index_type* mask) {
	ASSERT(mat && "Matrix is NULL");
	if(mat->op->add_elem_value_blocked_batched) {
		mat->op->add_elem_value_blocked_batched(mat, nshl,
																						batch_size, batch_index_ptr, ien,
																						block_row, block_col,
																						val, lda, stride, mask);
	}
}

// void MatrixAddElementLHS(Matrix* mat, index_type nshl, index_type bs,
// 												 index_type batch_size, const index_type* batch_ptr, const index_type* ien,
// 												 const value_type* val, int lda) {
// 	ASSERT(mat && "Matrix is NULL");
// 	if(mat->op->add_element_lhs) {
// 		mat->op->add_element_lhs(mat, nshl, bs,
// 														 batch_size, batch_ptr, ien,
// 														 val, lda);
// 	}
// }

void MatrixAddValueBatched(Matrix* mat,
													 index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
													 const value_type* A) {
	ASSERT(mat && "Matrix is NULL");
	if(mat->op->add_value_batched) {
		mat->op->add_value_batched(mat, batch_size, batch_row_ind, batch_col_ind, A);
	}
}

void MatrixAddValueBlockedBatched(Matrix* mat,
																  index_type batch_size,
																  const index_type* batch_row_ind, const index_type* batch_col_ind,
																  index_type block_row, index_type block_col,
																  const value_type* A, int lda, int stride) {
	ASSERT(mat && "Matrix is NULL");
	if(mat->op->add_value_blocked_batched) {
		mat->op->add_value_blocked_batched(mat, batch_size, batch_row_ind, batch_col_ind,
																			 block_row, block_col,
																			 A, lda, stride);
	}
}

__END_DECLS__
