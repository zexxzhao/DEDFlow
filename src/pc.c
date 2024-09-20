#include <string.h>
#include "vec.h"
#include "alloc.h"
#include "pc.h"

__BEGIN_DECLS__
struct CdamPCNone {
	index_type n;
	value_type w;
};


struct CdamPCJacobi {
	index_type n;
	index_type bs;
	void* diag;
};

struct CdamPCDecomposition {
	index_type n_sec;
	index_type* offset;
	CdamPC** pc;
};

struct CdamPCAMGX {
#ifdef USE_AMGX
	AMGX_config_handle cfg;
	AMGX_matrix_handle A;
	AMGX_vector_handle x;
	AMGX_vector_handle y;
	AMGX_solver_handle solver;
	AMGX_resources_handle rsrc;
	AMGX_Mode mode;
#endif
};

struct CdamPCCustom {
	void* ctx;
};


/* PCXXXAlloc allocte the space on memory */
/* PCXXXSetup fill the space with the data */
/* PCXXXApply apply the preconditioner */
/* PCXXXDestroy free the space */

static void PCNoneAlloc(PC* pc, index_type n) {
	PCNone* pc_impl = CdamTMalloc(PCNone, 1, HOST_MEM);
	CdamMemset(pc_impl, 0, sizeof(PCNone), HOST_MEM);
	pc->data = (void*)pc_impl;
	pc_impl->n = n;
}

static void PCNoneSetup(PC* pc) {
}

static void PCNoneApply(PC* pc, value_type* x, value_type* y) {
	PCNone* pc_impl = (PCNone*)pc->data;
	index_type n = pc_impl->n;
	CdamMemcpy(y, x, n * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
}

static void PCNoneDestroy(PC* pc) {
	PCNone* pc_impl = (PCNone*)pc->data;
	CdamFree(pc_impl, sizeof(PCNone), HOST_MEM);
}

static void PCJacobiAlloc(PC* pc, index_type bs, void* cublas_handle) {
	PCJacobi* pc_impl = CdamTMalloc(PCJacobi, 1, HOST_MEM);
	CdamMemset(pc_impl, 0, sizeof(PCJacobi), HOST_MEM);
	pc->cublas_handle = cublas_handle; 
	pc->data = (void*)pc_impl;
	pc_impl->n = MatrixNumRow((Matrix*)pc->mat);
	pc_impl->bs = bs;
	pc_impl->diag = CdamTMalloc(value_type, pc_impl->n * bs, DEVICE_MEM);
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
		inv = CdamTMalloc(value_type, num_node * bs * bs, DEVICE_MEM);
		input_batch = CdamTMalloc(value_type*, num_node * 2, DEVICE_MEM);
		output_batch = input_batch + num_node;
		info = CdamTMalloc(int, num_node * (bs + 1), DEVICE_MEM);
		pivot = info + num_node;

		h_batch = CdamTMalloc(value_type*, num_node, HOST_MEM);
		for(int i = 0; i < num_node; i++) {
			h_batch[i] = pc_ctx + i * bs * bs;
		}
		CdamMemcpy(input_batch, h_batch, num_node * sizeof(value_type*), DEVICE_MEM, HOST_MEM);
		for(int i = 0; i < num_node; i++) {
			h_batch[i] = inv + i * bs * bs;
		}
		CdamMemcpy(output_batch, h_batch, num_node * sizeof(value_type*), DEVICE_MEM, HOST_MEM);
		cublasDgetrfBatched(cublas_handle, bs, input_batch, bs, pivot, info, num_node);
		cublasDgetriBatched(cublas_handle, bs, (const value_type* const*)input_batch, bs,
												pivot, output_batch, bs, info, num_node);
		CdamMemcpy(pc_ctx, inv, num_node * bs * bs * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);


		CdamFree(inv, num_node * bs * bs * sizeof(value_type), DEVICE_MEM);
		CdamFree(input_batch, num_node * sizeof(value_type*) * 2, DEVICE_MEM);
		CdamFree(info, num_node * sizeof(int) * (bs + 1), DEVICE_MEM);
		CdamFree(h_batch, num_node * sizeof(value_type*), HOST_MEM);
	}
}

static void PCJacobiDestory(PC* pc) {
	PCJacobi* pc_impl = (PCJacobi*)pc->data;
	CdamFree(pc_impl->diag, pc_impl->n * pc_impl->bs * sizeof(value_type), DEVICE_MEM);
	CdamFree(pc_impl, sizeof(PCJacobi), HOST_MEM);
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
	PCDecomposition* pc_impl = CdamTMalloc(PCDecomposition, 1, HOST_MEM);
	CdamMemset(pc_impl, 0, sizeof(PCDecomposition), HOST_MEM);
	pc->cublas_handle = cublas_handle;
	pc->data = (void*)pc_impl;
	pc_impl->n_sec = n_sec;
	pc_impl->offset = CdamTMalloc(index_type, n_sec + 1, HOST_MEM);
	memcpy(pc_impl->offset, offset, sizeof(index_type) * (n_sec + 1));
	pc_impl->pc = CdamTMalloc(PC*, n_sec, HOST_MEM);
	CdamMemset(pc_impl->pc, 0, sizeof(PC*) * n_sec, HOST_MEM);
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
	CdamFreeHost(pc_impl->offset, sizeof(index_type) * (pc_impl->n_sec + 1));
	CdamFreeHost(pc_impl->pc, sizeof(PC*) * pc_impl->n_sec);
	CdamFreeHost(pc_impl, sizeof(PCDecomposition));
}

static void PCAMGXAlloc(PC* pc, void* options) {
#ifdef USE_AMGX
	Matrix* mat = (Matrix*)pc->mat;
	MatrixCSR* mat_csr = NULL;
	PCAMGX* pc_impl = CdamTMalloc(PCAMGX, 1, HOST_MEM);
	CdamMemset(pc_impl, 0, sizeof(PCAMGX), HOST_MEM);
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
	CdamFreeHost(pc_impl, sizeof(PCAMGX));
#endif
}

PC* PCCreateNone(Matrix* mat, index_type n) {
	PC* pc = CdamTMalloc(PC, 1, HOST_MEM);
	CdamMemset(pc, 0, sizeof(PC), HOST_MEM);
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
	PC* pc = CdamTMalloc(PC, 1, HOST_MEM);
	CdamMemset(pc, 0, sizeof(PC), HOST_MEM);
	pc->mat = (void*)mat;
	PCJacobiAlloc(pc, bs, cublas_handle);

	pc->op->setup = PCJacobiSetup;
	pc->op->apply = PCJacobiApply;
	pc->op->destroy = PCJacobiDestory;

	return pc;
}

PC* PCCreateDecomposition(Matrix* mat, index_type bs, const index_type* offset, void* cublas_handle) {
	PC* pc = CdamTMalloc(PC, 1, HOST_MEM);
	CdamMemset(pc, 0, sizeof(PC), HOST_MEM);
	pc->mat = (void*)mat;
	PCDecompositionAlloc(pc, bs, offset, cublas_handle);

	pc->op->setup = PCDecompositionSetup;
	pc->op->apply = PCDecompositionApply;
	pc->op->destroy = PCDecompositionDestroy;

	return pc;
}

PC* PCCreateAMGX(Matrix* mat, void* options) {
#ifdef USE_AMGX
	PC* pc = CdamTMalloc(PC, 1, HOST_MEM);
	CdamMemset(pc, 0, sizeof(PC), HOST_MEM);
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

void PCSetup(PC* pc, void* config) {
	
	if(strcmp(JSONGetItem(config, "Type")->valuestring, "Jacobi") == 0) {
		/* Seek "bs" */

		PCJacobiSetup(pc);
	}
	else if(strcmp(JSONGetItem(config, "Type")->valuestring, "Decomposition") == 0) {
		PCDecompositionSetup(pc);
	}
	else if(strcmp(JSONGetItem(config, "Type")->valuestring, "AMGX") == 0) {
		PCAMGXSetup(pc);
	}
	else {
		pc->op->setup(pc);
	}

	pc->op->setup(pc);
}

void PCDestroy(PC* pc) {
	if(pc == NULL) {
		return;
	}
	pc->op->destroy(pc);
	CdamFreeHost(pc, sizeof(PC));
}

#define streq_c(a, b) (strncmp(a, b, sizeof(b) - 1) == 0)

static void PCBuildPrivate(CdamPC* pc, void* A, void* config) {
	Matrix* mat = (Matrix*)A;
	cJSON* json = (cJSON*)config;

	CdamMemset(pc, 0, sizeof(CdamPC), HOST_MEM);
	pc->mat = A;

	pc->displ = JSONGetItem(json, "Displ")->valueint;
	pc->count = JSONGetItem(json, "Count")->valueint;

	if(streq_c(JSONGetItem(json, "Type")->valuestring, "Richardson")) {
		pc->type = PC_TYPE_RICHARDSON;
		pc->omega = JSONGetItem(json, "omega")->valuedouble;
	}
	else if(streq_c(JSONGetItem(json, "Type")->valuestring, "Jacobi")) {
		pc->type = PC_TYPE_JACOBI;
		pc->bs = JSONGetItem(json, "bs")->valueint;
		int bs = pc->bs;
		pc->diag = CdamTMalloc(value_type, pc->count, DEVICE_MEM);
		MatrixGetDiag(mat, pc->diag, bs);
		if(bs == 1) {
			VecPointwiseInv(pc->diag, pc->count);
		}
		else if(bs > 1) {
			int num_node = pc->count / bs;
			value_type* inv = CdamTMalloc(value_type, num_node * bs * bs, DEVICE_MEM);
			value_type** input_batch = CdamTMalloc(value_type*, num_node * 2, DEVICE_MEM);
			value_type** output_batch = input_batch + num_node;
			int* info = CdamTMalloc(int, num_node * (bs + 1), DEVICE_MEM);
			int* pivot = info + num_node;

			value_type** h_batch = CdamTMalloc(value_type*, num_node, HOST_MEM);
			for(int i = 0; i < num_node; i++) {
				h_batch[i] = pc->diag + i * bs * bs;
			}
			CdamMemcpy(input_batch, h_batch, num_node * sizeof(value_type*), DEVICE_MEM, HOST_MEM);
			for(int i = 0; i < num_node; i++) {
				h_batch[i] = inv + i * bs * bs;
			}
			CdamMemcpy(output_batch, h_batch, num_node * sizeof(value_type*), DEVICE_MEM, HOST_MEM);
			dgetrfBatched(bs, input_batch, bs, pivot, info, num_node);
			dgetriBatched(bs, (const value_type* const*)input_batch, bs, pivot, output_batch, bs, info, num_node);
			CdamMemcpy(pc->diag, inv, num_node * bs * bs * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);


			CdamFree(inv, num_node * bs * bs * sizeof(value_type), DEVICE_MEM);
			CdamFree(input_batch, num_node * sizeof(value_type*) * 2, DEVICE_MEM);
			CdamFree(info, num_node * sizeof(int) * (bs + 1), DEVICE_MEM);
			CdamFree(h_batch, num_node * sizeof(value_type*), HOST_MEM);
		}
	}
	else if(streq_c(JSONGetItem(json, "Type")->valuestring, "FieldSplit")) {
		pc->type = PC_TYPE_DECOMPOSITION;
		if(streq_c(JSONGetItem(json, "FieldSplitType")->valuestring, "Additive")) {
			pc->dtype = PC_DECOMPOSITION_ADDITIVE;
		}
		else if(streq_c(JSONGetItem(json, "FieldSplitType")->valuestring, "Multiplicative")) {
			pc->dtype = PC_DECOMPOSITION_MULTIPLICATIVE;
		}
		else if(streq_c(JSONGetItem(json, "FieldSplitType")->valuestring, "SchurFull")) {
			pc->dtype = PC_DECOMPOSITION_SCHUR_FULL;
		}
		else if(streq_c(JSONGetItem(json, "FieldSplitType")->valuestring, "SchurDiag")) {
			pc->dtype = PC_DECOMPOSITION_SCHUR_DIAG;
		}
		else if(streq_c(JSONGetItem(json, "FieldSplitType")->valuestring, "SchurUpper")) {
			pc->dtype = PC_DECOMPOSITION_SCHUR_UPPER;
		}
		else {
			fprintf(stderr, "Unknown FieldSplitType\n");
			exit(1);
		}

	}
	else if(streq_c(JSONGetItem(json, "Type")->valuestring, "KSP")) {
		pc->type = PC_TYPE_KSP;
		/* TODO: Implement KSP */
	}
	else {
		fprintf(stderr, "Unknown PC type\n");
		exit(1);
	}

}


static void PCApplyPrivate(CdamPC* pc, value_type* x, value_type* y) {

	index_type displ = pc->displ;
	index_type count = pc->count;

	if(pc->type == PC_TYPE_RICHARDSON) {
		CdamMemcpy(y + displ, x + displ, count * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
		dscal(count, pc->omega, y, 1);
	}
	else if(pc->type == PC_TYPE_JACOBI) {
		int bs = pc->bs;
		if(bs == 1) {
			VecPointwiseMult(x + displ, pc->diag, y + displ, count);
		}
		else if(bs > 1) {
			dgemvStridedBatched(BLAS_N, bs, bs, 1.0,
													pc->diag, bs, bs * bs,
													x + displ, 1, bs,
													y + displ, 1, bs,
													count / bs);
		}
	}
	else if(pc->type == PC_TYPE_DECOMPOSITION) {
		if(pc->dtype == PC_DECOMPOSITION_ADDITIVE) {
			CdamPC* pc_head = pc->child;
			while(pc_head) {
				PCApplyPrivate(pc_head, x, pc_head->tmp);
				pc_head = pc_head->next;
			}
			CdamMemset(y + displ, 0, count * sizeof(value_type), DEVICE_MEM);
			pc_head = pc->child;
			while(pc_head) {
				daxpy(count, 1.0, pc_head->tmp, 1, y + displ, 1);
				pc_head = pc_head->next;
			}
		}
		else if(pc->dtype = PC_DECOMPOSITION_MULTIPLICATIVE) {
			CdamPC* pc_head = pc->child;
			CdamMemcpy(pc_head->tmp, x, pc_head->n * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
			while(pc_head) {
				PCApplyPrivate(pc_head, pc_head->tmp, y);
				pc_head = pc_head->next;
				if(pc_head) {
					CdamMemcpy(pc_head->tmp, y, pc_head->n * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
				}
			}
		}
		else if(pc->dtype = PC_DECOMPOSITION_SCHUR_FULL
						|| pc->dtype == PC_DECOMPOSITION_SCHUR_DIAG
						|| pc->dtype == PC_DECOMPOSITION_SCHUR_UPPER) {
			/* Solve [I inv{A}*B; O I] [y0, y1] = [x0, x1] */
			if(pc->dtype & PC_DECOMPOSITION_SCHUR_UPPER) {
				CdamMemcpy(pc->tmp, x, pc->n * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
				/* TODO: pc->tmp[displ:displ+count] -= Ap * B * pc->tmp[displ1:displ1+count1] */

			}
			/* Solve [A O; O S] [y0, y1] = [x0, x1] */
			/* TODO: tmp[displ:displ+count] *= inv{Ap} */
			/* TODO: Implement Schur complement */
			/* TODO: tmp[displ1+displ1+count1] *= inv{S} */

			/* Solve [I O; -C*inv{A} I] [y0, y1] = [x0, x1]*/
			if(pc->dtype & PC_DECOMPOSITION_SCHUR_LOWER) {
			}
		}
	}
	else if(pc->type == PC_TYPE_KSP) {
		CdamKrylovSolve((CdamKrylov*)pc->ksp, (Matrix*)pc->mat, y + displ, x + displ);
	}
	else if(pc->type == PC_TYPE_CUSTOM) {
		pc->op[0]->apply(pc, x, y);
	}

	if(pc->next) {
		PCApplyPrivate(pc->next, x, y);
	}
}

__END_DECLS__

