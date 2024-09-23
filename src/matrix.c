#include <string.h>
#include "alloc.h"
#include "vec.h"
#include "matrix.h"
#include "matrix_impl.h"

__BEGIN_DECLS__

static void MaskVec(value_type* input, value_type* mask, value_type* output, int n) {
	VecPointwiseMult(input, mask, output, n);
} 

/****************************************************
 * CdamMatDense Operation
 ****************************************************/
static void CdamMatDestroyDensePrivate(CdamMat* mat) {
	index_type num_row = 0, num_col = 0;
	num_row += CdamMatNumExclusiveRow(mat);
	num_row += CdamMatNumSharedRow(mat);
	num_row += CdamMatNumGhostedRow(mat);
	num_col += CdamMatNumExclusiveCol(mat);
	num_col += CdamMatNumSharedCol(mat);
	num_col += CdamMatNumGhostedCol(mat);

	CdamFree(AsMatrixType(mat, CdamMatDense)->val, num_row * num_col * sizeof(value_type), DEVICE_MEM);
	CdamFree(CdamMatImpl(mat), sizeof(CdamMatDense), HOST_MEM);
	CdamFree(mat, sizeof(CdamMat), HOST_MEM);
}

static void CdamMatSetupDensePrivate(CdamMat* mat) {
	CdamMatDense* mat_dense = AsMatrixType(mat, CdamMatDense);
	index_type num_row = 0, num_col = 0;
	num_row += CdamMatNumExclusiveRow(mat);
	num_row += CdamMatNumSharedRow(mat);
	num_row += CdamMatNumGhostedRow(mat);
	num_col += CdamMatNumExclusiveCol(mat);
	num_col += CdamMatNumSharedCol(mat);
	num_col += CdamMatNumGhostedCol(mat);

}

CdamMatDense

/****************************************************
 * CdamMatCSR Operation
 ****************************************************/
CdamMatCSR* CdamMatCSRCreate(const CSRAttr* attr, void* ctx) {
	CdamMatCSR* mat = (CdamMatCSR*)CdamMallocHost(SIZE_OF(CdamMatCSR));
	mat->attr = attr;
	mat->val = (value_type*)CdamMallocDevice(CSRAttrNNZ(attr) * SIZE_OF(value_type));
	CUGUARD(cudaGetLastError());
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

void CdamMatCSRDestroy(CdamMat* mat) {
	CdamMatCSR* mat_csr = (CdamMatCSR*)mat->data;
	cusparseDestroySpMat(mat_csr->descr);
	index_type nnz = CSRAttrNNZ(mat_csr->attr);
	CdamFreeDevice(mat_csr->val, nnz * SIZE_OF(value_type));
	CdamFreeDevice(mat_csr->buffer, mat_csr->buffer_size);
	CdamFreeHost(mat, SIZE_OF(CdamMatCSR));
}

static void CdamMatCSRSetup(CdamMat* mat) {
}

/* Zero out the matrix */
static void CdamMatCSRZero(CdamMat* mat) {
	CdamMatCSR* mat_csr = (CdamMatCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	const index_type nnz = CSRAttrNNZ(attr);
	cudaMemsetAsync(mat_csr->val, 0, nnz * SIZE_OF(value_type), 0);
}

/* Zero the rows of the matrix */
static void CdamMatCSRZeroRow(CdamMat* mat, index_type n, const index_type* row, index_type shift, value_type diag) {
	CdamMatCSR* mat_csr = (CdamMatCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	const index_type num_row = CSRAttrNumRow(attr);
	const index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_idx = CSRAttrColInd(attr);
	value_type* val = mat_csr->val;

	CdamMatCSRZeroRowGPU(val,
											num_row, num_col, row_ptr, col_idx,
											n, row, shift, diag);
}


/* CdamMat-vector multiplication */
/* y = alpha * A * x + beta * y */
static void CdamMatCSRAMVPBY(CdamMat* A, value_type alpha, value_type* x, value_type beta, value_type* y) {
	CdamMatCSR* mat_csr = (CdamMatCSR*)A->data;
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
	cusparseHandle_t handle = *(cusparseHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUSPARSE_HANDLE);

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
			alg,
			&buffer_size
		);
		mat_csr->buffer_size = (index_type)buffer_size;
		mat_csr->buffer = (void*)CdamMallocDevice(mat_csr->buffer_size);
		/* If cuda version is 12.4 or higher, use cusparseSpMV_preprocess */ 
#if CUDART_VERSION >= 12040
		cusparseSpMV_preprocess(
			handle,
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
		handle,
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
static void CdamMatCSRAMVPBYWithMask(CdamMat* A, value_type alpha, value_type* x, value_type beta, value_type* y,
																		value_type* left_mask, value_type* right_mask) {
	CdamMatCSR* mat_csr = (CdamMatCSR*)A->data;
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
	CdamMatCSRAMVPBY(A, alpha, input, beta, y);

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
static void CdamMatCSRMatVec(CdamMat* mat, value_type* x, value_type* y) {
	value_type alpha = 1.0, beta = 0.0;
	CdamMatCSRAMVPBY(mat, alpha, x, beta, y);
}

/* y = left_mask(A * right_mask(x)) */
static void CdamMatCSRMatVecWithMask(CdamMat* mat, value_type* x, value_type* y,
														 value_type* left_mask, value_type* right_mask) {
	value_type alpha = 1.0, beta = 0.0;
	CdamMatCSRAMVPBYWithMask(mat, alpha, x, beta, y, left_mask, right_mask);
}

static void CdamMatCSRSubmatVec(CdamMat* mat, index_type i, index_type *ix,
															 index_type j, index_type *jx,
															 value_type* x, value_type* y) {
	ASSERT(0 && "Not implemented");
}

/* diag = diag(A) */
static void CdamMatCSRGetDiag(CdamMat* mat, value_type* diag, index_type bs) {
	CdamMatCSR* mat_csr = (CdamMatCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	const index_type num_row = CSRAttrNumRow(attr);
	const index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_idx = CSRAttrColInd(attr);
	const value_type* val = mat_csr->val;
	ASSERT(num_row == num_col && "CdamMat is not square");

	if(bs == 1) {
		CdamMatCSRGetDiagGPU(val, row_ptr, col_idx, diag, num_row);
	}
	else if(bs > 1) {
		if(attr->num_row != attr->parent->num_row * bs) {
			ASSERT(0 && "Block size is not compatible with the matrix size");
		}	
		attr = attr->parent;
		CdamMatGetDiagBlockGPU(mat_csr->val, bs,
													attr->num_row, attr->num_col, attr->row_ptr, attr->col_ind,
													diag, bs, bs * bs);
													
	}
	else {
		ASSERT(0 && "Block size should be greater than 0");
	}
}

static void CdamMatCSRSetValuesCOO(CdamMat* mat, value_type alpha,
																	index_type n, const index_type* row, const index_type* col,
																	const value_type* val, value_type beta) {
	CdamMatCSR* mat_csr = (CdamMatCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	index_type num_row = CSRAttrNumRow(attr);
	index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_ind = CSRAttrColInd(attr);

	CdamMatCSRSetValuesCOOGPU(mat_csr->val, alpha,
													 num_row, num_col, row_ptr, col_ind,
													 n, row, col, val, beta);
}

static void CdamMatCSRAddElemValueBatched(CdamMat* matrix, index_type nshl,
																				 index_type batch_size, const index_type* batch_ptr, const index_type* ien,
																				 const value_type* val, const index_type* mask) {
	CdamMatCSR* mat_csr = (CdamMatCSR*)matrix->data;
	const CSRAttr* attr = mat_csr->attr;
	index_type num_row = CSRAttrNumRow(attr);
	index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_ind = CSRAttrColInd(attr);

	CdamMatCSRAddElemValueBatchedGPU(mat_csr->val, 1.0,
																	batch_size, batch_ptr, ien, nshl,
																	num_row, num_col, row_ptr, col_ind,
																	val, 1.0, mask);
}

static void CdamMatCSRAddElemValueBlockedBatched(CdamMat* matrix, index_type nshl,
																								index_type batch_size, const index_type* batch_index_ptr, const index_type* ien,
																								index_type block_row, index_type block_col,
																								const value_type* val, int lda, int stride, const index_type* mask) {
	CdamMatCSR* mat_csr = (CdamMatCSR*)matrix->data;
	const CSRAttr* attr = mat_csr->attr;
	index_type num_row = CSRAttrNumRow(attr);
	index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_ind = CSRAttrColInd(attr);

	CdamMatCSRAddElemValueBlockedBatchedGPU(mat_csr->val, 1.0, 
																				 batch_size, batch_index_ptr, ien, nshl,
																				 num_row, num_col, row_ptr, col_ind,
																				 block_row, block_col,
																				 val, lda, stride, 1.0, mask);
}

static void CdamMatCSRAddElementLHS(CdamMat* mat, index_type nshl, index_type bs,
																	 index_type batch_size, const index_type* batch_index_ptr,
																	 const index_type* ien,
																	 const value_type* val, int lda) {
	CdamMatCSR* mat_csr = (CdamMatCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	index_type num_row = CSRAttrNumRow(attr);
	index_type num_col = CSRAttrNumCol(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_ind = CSRAttrColInd(attr);

	CdamMatCSRAddElementLHSGPU(mat_csr->val, nshl, bs,
														num_row, row_ptr, num_col, col_ind,
														batch_size, batch_index_ptr, ien,
														val, lda);
}

static void CdamMatCSRAddValueBatched(CdamMat* mat,
																		 index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
																		 const value_type* A) {
	CdamMatCSR* mat_csr = (CdamMatCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	index_type csr_num_row = CSRAttrNumRow(attr);
	index_type csr_num_col = CSRAttrNumCol(attr);
	const index_type* csr_row_ptr = CSRAttrRowPtr(attr);
	const index_type* csr_col_ind = CSRAttrColInd(attr);

	CdamMatCSRSetValueBatchedGPU(mat_csr->val, 1.0,
															csr_num_row, csr_num_col, csr_row_ptr, csr_col_ind, /* CSR */
															batch_size, batch_row_ind, batch_col_ind, /* Batched */
															A, 1.0);
}

static void CdamMatCSRAddValueBlockedBatched(CdamMat* mat,
																						index_type batch_size,
																						const index_type* batch_row_ind, const index_type* batch_col_ind,
																						index_type block_row, index_type block_col,
																						const value_type* A, int lda, int stride) {
	CdamMatCSR* mat_csr = (CdamMatCSR*)mat->data;
	const CSRAttr* attr = mat_csr->attr;
	index_type csr_num_row = CSRAttrNumRow(attr);
	index_type csr_num_col = CSRAttrNumCol(attr);
	const index_type* csr_row_ptr = CSRAttrRowPtr(attr);
	const index_type* csr_col_ind = CSRAttrColInd(attr);

	CdamMatCSRSetValueBlockedBatchedGPU(mat_csr->val, 1.0,
																			csr_num_row, csr_num_col, csr_row_ptr, csr_col_ind, /* CSR */
																			batch_size, batch_row_ind, batch_col_ind, /* Batched */
																			block_row, block_col, /* Block */
																			A, 1.0, lda, stride);
}



/****************************************************
 * CdamMatFS Operation
 ****************************************************/

CdamMatFS* CdamMatFSCreate(index_type n_offset, const index_type* offset, void* ctx) {
	CdamMatFS* mat = (CdamMatFS*)CdamMallocHost(SIZE_OF(CdamMatFS));
	memset(mat, 0, SIZE_OF(CdamMatFS));

	mat->n_offset = n_offset;

	mat->offset = (index_type*)CdamMallocHost(SIZE_OF(index_type) * (n_offset + 1));
	memcpy(mat->offset, offset, SIZE_OF(index_type) * (n_offset + 1));

	mat->d_offset = (index_type*)CdamMallocDevice((n_offset + 1) * SIZE_OF(index_type));
	cudaMemcpy(mat->d_offset, offset, (n_offset + 1) * SIZE_OF(index_type), cudaMemcpyHostToDevice);

	mat->d_matval = (value_type**)CdamMallocDevice(SIZE_OF(value_type*) * n_offset * n_offset);
	cudaMemset(mat->d_matval, 0, SIZE_OF(value_type*) * n_offset * n_offset);

	mat->mat = (CdamMat**)CdamMallocHost(SIZE_OF(CdamMat*) * n_offset * n_offset);
	memset(mat->mat, 0, SIZE_OF(CdamMat*) * n_offset * n_offset);

	mat->stream = (cudaStream_t*)CdamMallocHost(SIZE_OF(cudaStream_t) * n_offset);
	for(index_type i = 0; i < n_offset; i++) {
		cudaStreamCreate(mat->stream + i);
	}
	
	return mat;
}

void CdamMatFSDestroy(CdamMat* mat) {
	CdamMatFS* mat_fs = (CdamMatFS*)mat->data;
	index_type n = mat_fs->n_offset;
	for(index_type i = 0; i < n * n; i++) {
		if(mat_fs->mat[i]) {
			CdamMatDestroy(mat_fs->mat[i]);
		}
	}
	CdamFreeHost(mat_fs->offset, (n + 1) * SIZE_OF(index_type));
	CdamFreeDevice(mat_fs->d_offset, (n + 1) * SIZE_OF(index_type));
	CdamFreeDevice(mat_fs->d_matval, n * n * SIZE_OF(value_type*));
	CdamFreeHost(mat_fs->mat, n * n * SIZE_OF(CdamMat*));
	CdamFreeHost(mat_fs, SIZE_OF(CdamMatFS));
	CdamFreeHost(mat, SIZE_OF(CdamMat));

	for(index_type i = 0; i < n; i++) {
		cudaStreamDestroy(mat_fs->stream[i]);
	}
	CdamFreeHost(mat_fs->stream, SIZE_OF(cudaStream_t) * n);
}

static void CdamMatFSSetup(CdamMat* mat) {
	CdamMatFS* mat_fs = (CdamMatFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	index_type num_row = mat_fs->spy1x1->num_row;
	index_type num_col = mat_fs->spy1x1->num_col;

	CdamMat *submat = NULL;

	/* Set the matrix size */
	mat->size[0] = offset[n_offset] * num_row;
	mat->size[1] = offset[n_offset] * num_col;

	/* Set up the submatrices */
	for(index_type i = 0; i < n_offset * n_offset; i++) {
		if(mat_fs->mat[i]) {
			CdamMatSetup(mat_fs->mat[i]);
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
		matval[i] = ((CdamMatCSR*)submat->data)->val;
	}
	cudaMemcpy(mat_fs->d_matval, matval, SIZE_OF(value_type*) * n_offset * n_offset, cudaMemcpyHostToDevice);
	CdamFreeHost(matval, SIZE_OF(value_type*) * n_offset * n_offset);
}

static void CdamMatFSZero(CdamMat* mat) {
	CdamMatFS* mat_fs = (CdamMatFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	CdamMat** mat_list = mat_fs->mat;

	/* Zero out the matrix */
	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			CdamMatZero(mat_list[i * n_offset + j]);
		}
	}
}

static void CdamMatFSZeroRow(CdamMat* mat, index_type n, const index_type* row, index_type shift, value_type diag) {
	CdamMatFS* mat_fs = (CdamMatFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	index_type num_row = mat_fs->spy1x1->num_row;
	CdamMat** mat_list = mat_fs->mat;

	// ASSERT(0 && "Not implemented");

	/* Zero out the matrix */
	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			CdamMatZeroRow(mat_list[i * n_offset + j], n - offset[i] * num_row, row + offset[i] * num_row,
										-num_row * offset[i],
										i == j ? diag: 0.0 );
		}
	}
}

static void CdamMatFSAMVPBY(CdamMat* A, value_type alpha, value_type* x, value_type beta, value_type* y) {
	CdamMatFS* mat_fs = (CdamMatFS*)A->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	index_type num_row = mat_fs->spy1x1->num_row;
	index_type num_col = mat_fs->spy1x1->num_col;
	CdamMat** mat_list = mat_fs->mat;

	cublasHandle_t handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE);
	cublasDscal(handle, n_offset * num_row, &beta, y, 1);


	/* Perform matrix-vector multiplication */
	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			mat_list[i * n_offset + j]->stream_ref = NULL; //mat_fs->stream[i];
			cudaStreamSynchronize(0);
			CdamMatAMVPBY(mat_list[i * n_offset + j], alpha, x + offset[j] * num_col, 1.0, y + offset[i] * num_row);
			cudaStreamSynchronize(0);
			mat_list[i * n_offset + j]->stream_ref = NULL;
		}
	}

}

static void CdamMatFSAMVPBYWithMask(CdamMat* A, value_type alpha, value_type* x, value_type beta, value_type* y,
																	 value_type* left_mask, value_type* right_mask) {
	CdamMatFS* mat_fs = (CdamMatFS*)A->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	index_type num_row = mat_fs->spy1x1->num_row;
	index_type num_col = mat_fs->spy1x1->num_col;
	CdamMat** mat_list = mat_fs->mat;
	value_type* lm = left_mask, *rm = right_mask;

	/* Perform matrix-vector multiplication */
	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			CdamMatAMVPBYWithMask(mat_list[i * n_offset + j], alpha, x + offset[j] * num_col, beta, y + offset[i] * num_row,
													 (lm ? lm + offset[i] * num_col : NULL), (rm ? rm + offset[j] * num_row : NULL));
		}
	}
}

static void CdamMatFSMatVec(CdamMat* mat, value_type* x, value_type* y) {
	value_type alpha = 1.0, beta = 0.0;
	CdamMatFSAMVPBY(mat, alpha, x, beta, y);
}

static void CdamMatFSMatVecWithMask(CdamMat* mat, value_type* x, value_type* y,
																			 value_type* left_mask, value_type* right_mask) {
	value_type alpha = 1.0, beta = 0.0;
	CdamMatFSAMVPBYWithMask(mat, alpha, x, beta, y, left_mask, right_mask);
}

static void CdamMatFSSubmatVec(CdamMat* mat, index_type i, index_type *ix,
															index_type j, index_type *jx,
															value_type *x, value_type *y) {
	CdamMatFS* mat_fs = (CdamMatFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	index_type num_row = mat_fs->spy1x1->num_row;
	index_type num_col = mat_fs->spy1x1->num_col;
	index_type ik = 0, jk = 0;
	index_type bs;
	value_type *ytmp, *xtmp;

	CdamMat** mat_list = mat_fs->mat;

	/* Zero out y[:] */

	bs = 0;
	ytmp = y;
	for(ik = 0; ik < j; ik++) {
		bs = offset[ix[ik + 1]] - offset[ix[ik]];
		ytmp += bs * num_row;
		CdamMemset(ytmp, 0, num_row * bs * sizeof(value_type));
	}


	ytmp = y;
	for(ik = 0; ik < i; ++ik) {
		ytmp += (offset[ix[ik + 1]] - offset[ix[ik]]) * num_row;

		xtmp = x;
		for(jk = 0; jk < j; ++jk) {
			if(mat_list[ix[ik] * n_offset + jx[jk]] == NULL) {
				continue;
			}
			xtmp += (offset[jx[jk + 1]] - offset[jx[jk]]) * num_col;
			CdamMatMatVec(mat_list[ix[ik] * n_offset + jx[jk]], xtmp, ytmp);
		}
	}
}


static void CdamMatFSGetDiag(CdamMat* mat, value_type* diag, index_type bs) {
	CdamMatFS* mat_fs = (CdamMatFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	index_type num_row = mat_fs->spy1x1->num_row;
	CdamMat** mat_list = mat_fs->mat;
	
	/* Set to zero */
	cudaMemset(diag, 0, SIZE_OF(value_type) * offset[n_offset] * num_row);
	/* Get diagonal elements */
	for(index_type i = 0; i < n_offset; i++) {
		if(mat_list[i * n_offset + i] == NULL) {
			continue;
		}
		CdamMatGetDiag(mat_list[i * n_offset + i], diag + offset[i] * num_row, 1);
	}


	// CdamFreeDevice(nzind, batch_size * SIZE_OF(index_type));
}

static void CdamMatFSSetValuesCOO(CdamMat* mat, value_type alpha,
																		 index_type n, const index_type* row, const index_type* col,
																		 const value_type* val, value_type beta) {
	CdamMatFS* mat_fs = (CdamMatFS*)mat->data;
	ASSERT(0 && "Not implemented");
}
static void CdamMatFSSetValuesInd(CdamMat* mat, value_type alpha,
																		 index_type n, const index_type* ind,
																		 const value_type* val, value_type beta) {
	CdamMatFS* mat_fs = (CdamMatFS*)mat->data;
	ASSERT(0 && "Not implemented");
}


static void CdamMatFSAddElemValueBatched(CdamMat* mat, index_type nshl,
																						index_type batch_size, const index_type* batch_index_ptr,
																						const index_type* ien, const value_type* val, const index_type* mask) {
	ASSERT(0 && "Not supported yet");

}

static void CdamMatFSAddElemValueBlockedBatched(CdamMat* mat, index_type nshl,
																							 index_type batch_size, const index_type* batch_index_ptr,
																							 const index_type* ien,
																							 index_type block_row, index_type block_col,
																							 const value_type* val, int lda, int stride, const index_type* mask) {
	CdamMatFS* mat_fs = (CdamMatFS*)mat->data;
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
																						

static void CdamMatFSAddElementLHS(CdamMat* mat, index_type nshl, index_type bs,
																	index_type batch_size, const index_type* batch_index_ptr, const index_type* ien,
																			const value_type* val, int lda) {
	CdamMatFS* mat_fs = (CdamMatFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	CdamMat** mat_list = mat_fs->mat;
	ASSERT(0 && "Not implemented yet");

	// CdamMatFSAddElementLHSGPU(mat_list, n_offset, offset, nshl, bs,
	// 														 batch_size, batch_index_ptr, ien,
	// 														 val, lda);
}

static void CdamMatFSAddValueBatched(CdamMat* mat,
																		index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
																		const value_type* A, int lda, int stride) {

	ASSERT(0 && "Not implemented yet");
}

static void CdamMatFSAddValueBlockedBatched(CdamMat* mat,
																					 index_type batch_size,
																					 const index_type* batch_row_ind, const index_type* batch_col_ind,
																					 index_type block_row, index_type block_col,
																					 const value_type* A, int lda, int stride) {
	CdamMatFS* mat_fs = (CdamMatFS*)mat->data;
	index_type n_offset = mat_fs->n_offset;
	const index_type* offset = mat_fs->offset;
	CdamMat** mat_list = mat_fs->mat;

	const CSRAttr* attr = mat_fs->spy1x1;
	ASSERT(attr && "CSRAttr is NULL");

	// index_type* nzind = (index_type*)CdamMallocDevice(batch_size * SIZE_OF(index_type));
	

	// CSRAttrGetNZIndBatched(attr, batch_size, batch_row_ind, batch_col_ind, nzind);

	for(index_type i = 0; i < n_offset; i++) {
		for(index_type j = 0; j < n_offset; j++) {
			if(mat_list[i * n_offset + j] == NULL) {
				continue;
			}
			CdamMatAddValueBlockedBatched(mat_list[i * n_offset + j],
																	 batch_size, batch_row_ind, batch_col_ind,
																	 offset[i + 1] - offset[i], offset[j + 1] - offset[j],
																	 A + offset[i] * lda + offset[j],
																	 lda, stride);
		}
	}


	// CdamFreeDevice(nzind, batch_size * SIZE_OF(index_type));
}

/****************************************************
 * General CdamMat Operation
 ****************************************************/
CdamMat* CdamMatCreateTypeCSR(const CSRAttr* attr, void* ctx) {
	CdamMat *mat = (CdamMat*)CdamMallocHost(SIZE_OF(CdamMat));
	mat->size[0] = CSRAttrNumRow(attr);
	mat->size[1] = CSRAttrNumCol(attr);

	/* Set up the matrix type */
	mat->type = MAT_TYPE_CSR;

	/* Set up the matrix data */	
	mat->data = CdamMatCSRCreate(attr, ctx);

	/* Set up the matrix operation */
	mat->op->setup = CdamMatCSRSetup;

	mat->op->zero = CdamMatCSRZero;
	mat->op->zero_row = CdamMatCSRZeroRow;

	mat->op->amvpby = CdamMatCSRAMVPBY;
	mat->op->amvpby_mask = CdamMatCSRAMVPBYWithMask;
	mat->op->matvec = CdamMatCSRMatVec;
	mat->op->matvec_mask = CdamMatCSRMatVecWithMask;
	mat->op->submatvec = CdamMatCSRSubmatVec;

	mat->op->get_diag = CdamMatCSRGetDiag;

	mat->op->set_values_coo = CdamMatCSRSetValuesCOO;
	mat->op->set_values_ind = NULL; /* CdamMatCSRSetValuesInd;*/

	mat->op->add_elem_value_batched = CdamMatCSRAddElemValueBatched;
	mat->op->add_elem_value_blocked_batched = CdamMatCSRAddElemValueBlockedBatched;

	mat->op->add_value_batched = NULL; /* CdamMatCSRAddValueBatched; */
	mat->op->add_value_blocked_batched = NULL; /* CdamMatCSRAddValueBlockedBatched; */

	mat->op->destroy = CdamMatCSRDestroy;
	return mat;
}



CdamMat* CdamMatCreateTypeFS(index_type n_offset, const index_type* offset, void* ctx) {
	CdamMat* mat = (CdamMat*)CdamMallocHost(SIZE_OF(CdamMat));
	mat->size[0] = offset[n_offset];
	mat->size[1] = offset[n_offset];

	/* Set up the matrix type */
	mat->type = MAT_TYPE_FS;

	/* Set up the matrix data */
	mat->data = CdamMatFSCreate(n_offset, offset, ctx);

	/* Set up the matrix operation */
	mat->op->setup = CdamMatFSSetup;

	mat->op->zero = CdamMatFSZero;	
	mat->op->zero_row = CdamMatFSZeroRow;

	mat->op->amvpby = CdamMatFSAMVPBY;
	mat->op->amvpby_mask = CdamMatFSAMVPBYWithMask;
	mat->op->matvec = CdamMatFSMatVec;
	mat->op->matvec_mask = CdamMatFSMatVecWithMask;
	mat->op->submatvec = CdamMatFSSubmatVec;

	mat->op->get_diag = CdamMatFSGetDiag;

	mat->op->set_values_coo = NULL; /* CdamMatFSSetValuesCOO; */
	mat->op->set_values_ind = NULL; /* CdamMatFSSetValuesInd; */

	mat->op->add_elem_value_batched = CdamMatFSAddElemValueBatched;
	mat->op->add_elem_value_blocked_batched = CdamMatFSAddElemValueBlockedBatched;

	mat->op->add_value_batched = NULL; /* CdamMatFSAddValueBatched; */
	mat->op->add_value_blocked_batched = CdamMatFSAddValueBlockedBatched;

	mat->op->destroy = CdamMatFSDestroy;
	return mat;
}

#define MATRIX_CALL(mat, func, ...) \
	do { \
		if(mat->op->func) { \
			mat->op->func(mat, ##__VA_ARGS__); \
		} \
		else { \
			fprintf(stderr, "CdamMat operation %s is not implemented for type: %d\n", #func, mat->type); \
		} \
	} while(0)

void CdamMatDestroy(CdamMat* mat) {
	if(mat == NULL) {
		return;
	}
	MATRIX_CALL(mat, destroy);
}

void CdamMatSetup(CdamMat* mat) {
	ASSERT(mat && "CdamMat is NULL");
	MATRIX_CALL(mat, setup);
}

void CdamMatZero(CdamMat* mat) {
	ASSERT(mat && "CdamMat is NULL");
	MATRIX_CALL(mat, zero);
}

void CdamMatZeroRow(CdamMat* mat, index_type n, const index_type* row, index_type shift, value_type diag) {
	ASSERT(mat && "CdamMat is NULL");
	MATRIX_CALL(mat, zero_row, n, row, shift, diag);
}

void CdamMatAMVPBY(CdamMat* A, value_type alpha, value_type* x, value_type beta, value_type* y) {
	ASSERT(A && "CdamMat is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	MATRIX_CALL(A, amvpby, alpha, x, beta, y);
}

void CdamMatAMVPBYWithMask(CdamMat* A, value_type alpha, value_type* x, value_type beta, value_type* y,
													value_type* left_mask, value_type* right_mask) {
	ASSERT(A && "CdamMat is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	MATRIX_CALL(A, amvpby_mask, alpha, x, beta, y, left_mask, right_mask);
}


void CdamMatMatVec(CdamMat* mat, value_type* x, value_type* y) {
	ASSERT(mat && "CdamMat is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	MATRIX_CALL(mat, matvec, x, y);
}

void CdamMatMatVecWithMask(CdamMat* mat, value_type* x, value_type* y, value_type* left_mask, value_type* right_mask) {
	ASSERT(mat && "CdamMat is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	MATRIX_CALL(mat, matvec_mask, x, y, left_mask, right_mask);
}

void CdamMatSubmatVec(CdamMat* mat, index_type i, index_type *ix,
										 index_type j, index_type *jx,
										 value_type *x, value_type *y) {
	ASSERT(mat && "CdamMat is NULL");
	ASSERT(ix && "ix is NULL");
	ASSERT(jx && "jx is NULL");
	ASSERT(x && "x is NULL");
	ASSERT(y && "y is NULL");
	MATRIX_CALL(mat, submatvec, i, ix, j, jx, x, y);
}

void CdamMatGetDiag(CdamMat* mat, value_type* diag, index_type bs) {
	ASSERT(mat && "CdamMat is NULL");
	ASSERT(diag && "diag is NULL");
	MATRIX_CALL(mat, get_diag, diag, bs);
}

void CdamMatSetValuesCOO(CdamMat* mat, value_type alpha,
											  index_type n, const index_type* row, const index_type* col,
											  const value_type* val, value_type beta) {
	ASSERT(mat && "CdamMat is NULL");
	MATRIX_CALL(mat, set_values_coo, alpha, n, row, col, val, beta);
}

void CdamMatSetValuesInd(CdamMat* mat, value_type alpha,
												index_type n, const index_type* ind,
												const value_type* val, value_type beta) {
	ASSERT(mat && "CdamMat is NULL");
	MATRIX_CALL(mat, set_values_ind, alpha, n, ind, val, beta);
}

void CdamMatAddElemValueBatched(CdamMat* mat, index_type nshl,
															 index_type batch_size, const index_type* batch_index_ptr,
															 const index_type* ien, const value_type* val, const index_type* mask) {
	ASSERT(mat && "CdamMat is NULL");
	MATRIX_CALL(mat, add_elem_value_batched, nshl, batch_size, batch_index_ptr, ien, val, mask);
}

void CdamMatAddElemValueBlockedBatched(CdamMat* mat, index_type nshl,
																			index_type batch_size, const index_type* batch_index_ptr,
																			const index_type* ien,
																			index_type block_row, index_type block_col,
																			const value_type* val, int lda, int stride, const index_type* mask) {
	ASSERT(mat && "CdamMat is NULL");
	if(mat->op->add_elem_value_blocked_batched) {
		mat->op->add_elem_value_blocked_batched(mat, nshl,
																						batch_size, batch_index_ptr, ien,
																						block_row, block_col,
																						val, lda, stride, mask);
	}
}

// void CdamMatAddElementLHS(CdamMat* mat, index_type nshl, index_type bs,
// 												 index_type batch_size, const index_type* batch_ptr, const index_type* ien,
// 												 const value_type* val, int lda) {
// 	ASSERT(mat && "CdamMat is NULL");
// 	if(mat->op->add_element_lhs) {
// 		mat->op->add_element_lhs(mat, nshl, bs,
// 														 batch_size, batch_ptr, ien,
// 														 val, lda);
// 	}
// }

void CdamMatAddValueBatched(CdamMat* mat,
													 index_type batch_size, const index_type* batch_row_ind, const index_type* batch_col_ind,
													 const value_type* A) {
	ASSERT(mat && "CdamMat is NULL");
	if(mat->op->add_value_batched) {
		mat->op->add_value_batched(mat, batch_size, batch_row_ind, batch_col_ind, A);
	}
}

void CdamMatAddValueBlockedBatched(CdamMat* mat,
																  index_type batch_size,
																  const index_type* batch_row_ind, const index_type* batch_col_ind,
																  index_type block_row, index_type block_col,
																  const value_type* A, int lda, int stride) {
	ASSERT(mat && "CdamMat is NULL");
	if(mat->op->add_value_blocked_batched) {
		mat->op->add_value_blocked_batched(mat, batch_size, batch_row_ind, batch_col_ind,
																			 block_row, block_col,
																			 A, lda, stride);
	}
}

void CdamMatCreate(CdamMat** mat, MatType type) {
	CdamMat* A = CdamTMalloc(CdamMat, 1, HOST_MEM);
	CdamMemset(A, sizeof(CdamMat), HOST_MEM);
	*mat = A;

	CdamMatType(A) = type;	
	if(type == MAT_TYPE_DENSE) {
		CdamMatImpl(A) = CdamTMalloc(CdamMatDense, 1, HOST_MEM);
		CdamMemset(CdamMatImpl(A), sizeof(CdamMatDense), HOST_MEM);
	}
	else if(type == MAT_TYPE_CSR) {
		CdamMatImpl(A) = CdamTMalloc(CdamMatCSR, 1, HOST_MEM);
		CdamMemset(CdamMatImpl(A), sizeof(CdamMatCSR), HOST_MEM);
	}
	else if(type == MAT_TYPE_BSR) {
		CdamMatImpl(A) = CdamTMalloc(CdamMatBSR, 1, HOST_MEM);
		CdamMemset(CdamMatImpl(A), sizeof(CdamMatBSR), HOST_MEM);
	}
}

void CdamMatDestroy(CdamMat* mat) {
	if(mat == NULL) {
		return;
	}
	CdamMatType type = CdamMatType(mat);	
	void* impl = CdamMatImpl(mat);
	if(type == MAT_TYPE_DENSE) {
		CdamMatDense* dense = (CdamMatDense*)impl;
		index_type num_row = 0, num_col = 0;
		num_row += CdamMatNumExclusiveRow(mat);
		num_row += CdamMatNumSharedRow(mat);
		num_row += CdamMatNumGhostedRow(mat);

		num_col += CdamMatNumExclusiveCol(mat);
		num_col += CdamMatNumSharedCol(mat);
		num_col += CdamMatNumGhostedCol(mat);

		CdamFree(dense->val, sizeof(value_type) * num_row * num_col, DEVICE_MEM);
		CdamFree(dense, sizeof(CdamMatDense), HOST_MEM);
	}
	else if(type == MAT_TYPE_CSR) {
		CdamMatCSR* csr = (CdamMatCSR*)impl;
		CdamMatCSRDestroy(csr);
	}
	else if(type == MAT_TYPE_BSR) {
		CdamMatBSR* bsr = (CdamMatBSR*)impl;
		CdamFree(bsr, HOST_MEM);
	}

	CdamFree(mat, 1, HOST_MEM);
}

__END_DECLS__
