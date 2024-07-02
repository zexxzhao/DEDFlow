#include <string.h>
#include <cublas_v2.h>
#include "vec.h"
#include "alloc.h"
#include "pc.h"

__BEGIN_DECLS__
/* PCXXXAlloc allocte the space on memory */
/* PCXXXSetup fill the space with the data */
/* PCXXXApply apply the preconditioner */
/* PCXXXDestroy free the space */

static void PCNoneAlloc(PC* pc, index_type n) {
	PCNone* pc_impl = (PCNone*)CdamMallocHost(SIZE_OF(PCNone));
	memset(pc_impl, 0, SIZE_OF(PCNone));
	pc->data = (void*)pc_impl;
	pc_impl->n = n;
}

static void PCNoneSetup(PC* pc) {
}

static void PCNoneApply(PC* pc, value_type* x, value_type* y) {
	PCNone* pc_impl = (PCNone*)pc->data;
	index_type n = pc_impl->n;
	cudaMemcpyAsync(y, x, n * SIZE_OF(value_type), cudaMemcpyDeviceToDevice, 0);
}

static void PCNoneDestroy(PC* pc) {
	PCNone* pc_impl = (PCNone*)pc->data;
	CdamFreeHost(pc_impl, SIZE_OF(PCNone));
}

static void PCJacobiAlloc(PC* pc, index_type bs, void* cublas_handle) {
	PCJacobi* pc_impl = (PCJacobi*)CdamMallocHost(SIZE_OF(PCJacobi));
	memset(pc_impl, 0, SIZE_OF(PCJacobi));
	pc->cublas_handle = cublas_handle; 
	pc->data = (void*)pc_impl;
	pc_impl->n = MatrixNumRow((Matrix*)pc->mat);
	pc_impl->bs = bs;
	pc_impl->diag = CdamMallocDevice(SIZE_OF(value_type) * pc_impl->n * bs);
}

static void PCJacobiSetup(PC* pc) {
	PCJacobi* pc_impl = (PCJacobi*)pc->data;
	Matrix* mat = (Matrix*)pc->mat;
	index_type num_row = MatrixNumRow(mat);
	index_type bs = pc_impl->bs;
	index_type num_node = num_row / bs;
	value_type* pc_ctx = (value_type*)pc_impl->diag;
	MatrixGetDiag(mat, pc_ctx, bs);
	value_type *inv, **input_batch, **output_batch, **h_batch;
	int* info, *pivot;
	cublasHandle_t cublas_handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE);

	if(bs == 1) {
		VecPointwiseInv(pc_ctx, num_row);
	}
	else if(bs > 1) {
		inv = (value_type*)CdamMallocDevice(num_node * bs * bs * SIZE_OF(value_type));
		input_batch = (value_type**)CdamMallocDevice(num_node * SIZE_OF(value_type*) * 2);
		output_batch = input_batch + num_node;
		info = (int*)CdamMallocDevice(num_node * SIZE_OF(int) * (bs + 1));
		pivot = info + num_node;

		h_batch = (value_type**)CdamMallocHost(num_node * SIZE_OF(value_type*));
		for(int i = 0; i < num_node; i++) {
			h_batch[i] = pc_ctx + i * bs * bs;
		}
		cudaMemcpy(input_batch, h_batch, num_node * SIZE_OF(value_type*), cudaMemcpyHostToDevice);
		for(int i = 0; i < num_node; i++) {
			h_batch[i] = inv + i * bs * bs;
		}
		cudaMemcpy(output_batch, h_batch, num_node * SIZE_OF(value_type*), cudaMemcpyHostToDevice);
		cublasDgetrfBatched(cublas_handle, bs, input_batch, bs, pivot, info, num_node);
		cublasDgetriBatched(cublas_handle, bs, (const value_type* const*)input_batch, bs,
												pivot, output_batch, bs, info, num_node);
		cudaMemcpy(pc_ctx, inv, num_node * bs * bs * SIZE_OF(value_type), cudaMemcpyDeviceToDevice);

		CdamFreeDevice(inv, num_node * bs * bs * SIZE_OF(value_type));
		CdamFreeDevice(input_batch, num_node * SIZE_OF(value_type*) * 2);
		CdamFreeDevice(info, num_node * SIZE_OF(int) * (bs + 1));
		CdamFreeHost(h_batch, num_node * SIZE_OF(value_type*));
	}
}

static void PCJacobiDestory(PC* pc) {
	PCJacobi* pc_impl = (PCJacobi*)pc->data;
	CdamFreeDevice(pc_impl->diag, pc_impl->n * pc_impl->bs * SIZE_OF(value_type));
	CdamFreeHost(pc_impl, SIZE_OF(PCJacobi));
}

static void PCJacobiApply(PC* pc, value_type* x, value_type* y) {
	PCJacobi* pc_impl = (PCJacobi*)pc->data;
	index_type bs = pc_impl->bs;
	value_type* diag = (value_type*)pc_impl->diag;
	cublasHandle_t cublas_handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE);
	value_type one = 1.0, zero = 0.0;

	if(bs == 1) {
		VecPointwiseMult(x, diag, y, pc_impl->n);
	}
	else if(bs > 1) {
		cublasDgemvStridedBatched(cublas_handle,
															CUBLAS_OP_N,
															bs, bs,
															&one,
															diag, bs, bs * bs,
															x, 1, bs,
															&zero,
															y, 1, bs,
															pc_impl->n / bs);
	}
}

static void PCDecompositionAlloc(PC* pc, index_type n_sec, const index_type* offset, void* cublas_handle) {
	PCDecomposition* pc_impl = (PCDecomposition*)CdamMallocHost(SIZE_OF(PCDecomposition));
	memset(pc_impl, 0, SIZE_OF(PCDecomposition));
	pc->cublas_handle = cublas_handle;
	pc->data = (void*)pc_impl;
	pc_impl->n_sec = n_sec;
	pc_impl->offset = (index_type*)CdamMallocHost(SIZE_OF(index_type) * (pc_impl->n_sec + 1));
	memcpy(pc_impl->offset, offset, SIZE_OF(index_type) * (n_sec + 1));
	pc_impl->pc = (PC**)CdamMallocHost(SIZE_OF(PC*) * n_sec);
	memset(pc_impl->pc, 0, SIZE_OF(PC*) * n_sec);
}

static void PCDecompositionSetup(PC* pc) {
	PCDecomposition* pc_impl = (PCDecomposition*)pc->data;
	index_type i;
	for(i = 0; i < pc_impl->n_sec; i++) {
		PCSetup(pc_impl->pc[i]);
	}	
}

static void PCDecompositionApply(PC* pc, value_type* x, value_type* y) {
	PCDecomposition* pc_impl = (PCDecomposition*)pc->data;
	index_type i, n_sec = pc_impl->n_sec;
	index_type* offset = pc_impl->offset;
	PC** pc_sec = pc_impl->pc;
	value_type* x_sec, *y_sec;
	for(i = 0; i < n_sec; i++) {
		x_sec = x + offset[i];
		y_sec = y + offset[i];
		PCApply(pc_sec[i], x_sec, y_sec);
	}
} 

static void PCDecompositionDestroy(PC* pc) {
	PCDecomposition* pc_impl = (PCDecomposition*)pc->data;
	index_type i;
	for(i = 0; i < pc_impl->n_sec; i++) {
		PCDestroy(pc_impl->pc[i]);
	}
	CdamFreeHost(pc_impl->offset, SIZE_OF(index_type) * (pc_impl->n_sec + 1));
	CdamFreeHost(pc_impl->pc, SIZE_OF(PC*) * pc_impl->n_sec);
	CdamFreeHost(pc_impl, SIZE_OF(PCDecomposition));
}

static void PCAMGXAlloc(PC* pc, void* options) {
#ifdef USE_AMGX
	Matrix* mat = (Matrix*)pc->mat;
	MatrixCSR* mat_csr = NULL;
	PCAMGX* pc_impl = (PCAMGX*)CdamMallocHost(SIZE_OF(PCAMGX));
	memset(pc_impl, 0, SIZE_OF(PCAMGX));
	pc_impl->mode = AMGX_mode_dDDI;
	if(mat->type != MAT_TYPE_CSR) {
		fprintf(stderr, "Matrix type is not CSR\n");
		exit(1);
	}
	mat_csr = (MatrixCSR*)mat->data;


	// AMGX_config_create(&(pc_impl->cfg), (const char*)options);
	AMGX_config_create_from_file(&(pc_impl->cfg), (const char*)options);
	AMGX_resources_create_simple(&(pc_impl->rsrc), pc_impl->cfg);
	AMGX_matrix_create(&(pc_impl->A), pc_impl->rsrc, pc_impl->mode);
	AMGX_vector_create(&(pc_impl->x), pc_impl->rsrc, pc_impl->mode);
	AMGX_vector_create(&(pc_impl->y), pc_impl->rsrc, pc_impl->mode);
	AMGX_solver_create(&(pc_impl->solver), pc_impl->rsrc, pc_impl->mode, pc_impl->cfg);
	AMGX_matrix_upload_all(pc_impl->A,
												 mat_csr->attr->num_row, 
												 mat_csr->attr->nnz,
												 1, 1,
												 mat_csr->attr->row_ptr,
												 mat_csr->attr->col_ind,
												 mat_csr->val,
												 NULL);

	AMGX_solver_setup(pc_impl->solver, pc_impl->A);
	pc->data = (void*)pc_impl;
#endif
}

static void PCAMGXSetup(PC* pc) {
#ifdef USE_AMGX
	Matrix* mat = (Matrix*)pc->mat;
	MatrixCSR* mat_csr = (MatrixCSR*)mat->data;
	PCAMGX* pc_impl = (PCAMGX*)pc->data;

	/* AMGX_matrix_handle is not set */
	AMGX_matrix_replace_coefficients(pc_impl->A,
																	 mat_csr->attr->num_row,
																	 mat_csr->attr->nnz,
																	 mat_csr->val,
																	 NULL);

	AMGX_solver_resetup(pc_impl->solver, pc_impl->A);
#endif
}

static void PCAMGXApply(PC* pc, value_type* x, value_type* y) {
#ifdef USE_AMGX
	index_type num_row = MatrixNumRow((Matrix*)pc->mat);
	PCAMGX* pc_impl = (PCAMGX*)pc->data;
	AMGX_vector_upload(pc_impl->x, num_row, 1, x);
	AMGX_vector_upload(pc_impl->y, num_row, 1, y);
	/// AMGX_vector_set_zero(pc_impl->y);
	AMGX_solver_solve_with_0_initial_guess(pc_impl->solver, pc_impl->x, pc_impl->y);
	AMGX_vector_download(pc_impl->y, y);
#endif
}

static void PCAMGXDestroy(PC* pc) {
#ifdef USE_AMGX
	PCAMGX* pc_impl = (PCAMGX*)pc->data;
	AMGX_solver_destroy(pc_impl->solver);
	AMGX_config_destroy(pc_impl->cfg);
	AMGX_vector_destroy(pc_impl->x);
	AMGX_vector_destroy(pc_impl->y);
	AMGX_matrix_destroy(pc_impl->A);
	AMGX_resources_destroy(pc_impl->rsrc);
	CdamFreeHost(pc_impl, SIZE_OF(PCAMGX));
#endif
}

PC* PCCreateNone(Matrix* mat, index_type n) {
	PC* pc = (PC*)CdamMallocHost(SIZE_OF(PC));
	memset(pc, 0, SIZE_OF(PC));
	if(mat) {
		n = MatrixNumRow(mat);
	}
	pc->mat = (void*)mat;
	PCNoneAlloc(pc, n);

	pc->op->setup = PCNoneSetup;
	pc->op->apply = PCNoneApply;
	pc->op->destroy = PCNoneDestroy;

	return pc;
}

PC* PCCreateJacobi(Matrix* mat, index_type bs, void* cublas_handle) {
	PC* pc = (PC*)CdamMallocHost(SIZE_OF(PC));
	memset(pc, 0, SIZE_OF(PC));
	pc->mat = (void*)mat;
	PCJacobiAlloc(pc, bs, cublas_handle);

	pc->op->setup = PCJacobiSetup;
	pc->op->apply = PCJacobiApply;
	pc->op->destroy = PCJacobiDestory;

	return pc;
}

PC* PCCreateDecomposition(Matrix* mat, index_type bs, const index_type* offset, void* cublas_handle) {
	PC* pc = (PC*)CdamMallocHost(SIZE_OF(PC));
	memset(pc, 0, SIZE_OF(PC));
	pc->mat = (void*)mat;
	PCDecompositionAlloc(pc, bs, offset, cublas_handle);

	pc->op->setup = PCDecompositionSetup;
	pc->op->apply = PCDecompositionApply;
	pc->op->destroy = PCDecompositionDestroy;

	return pc;
}

PC* PCCreateAMGX(Matrix* mat, void* options) {
#ifdef USE_AMGX
	PC* pc = (PC*)CdamMallocHost(SIZE_OF(PC));
	memset(pc, 0, SIZE_OF(PC));
	pc->mat = (void*)mat;
	PCAMGXAlloc(pc, options);

	pc->op->setup = PCAMGXSetup;
	pc->op->apply = PCAMGXApply;
	pc->op->destroy = PCAMGXDestroy;

	return pc;
#else
#warning "AMGX is not enabled"
	return NULL;
#endif
}

void PCApply(PC* pc, value_type* x, value_type* y) {
	pc->op->apply(pc, x, y);
}

void PCSetup(PC* pc) {
	pc->op->setup(pc);
}

void PCDestroy(PC* pc) {
	if(pc == NULL) {
		return;
	}
	pc->op->destroy(pc);
	CdamFreeHost(pc, SIZE_OF(PC));
}

__END_DECLS__
