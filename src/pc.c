#include <string.h>
#include "json.h"
#include "layout.h"
#include "vec_impl.h"
#include "blas.h"
#include "parallel_matrix.h"
#include "alloc.h"
#include "krylov.h"
#include "pc.h"

__BEGIN_DECLS__
#define streq_c(a, b) (strncmp(a, b, sizeof(b) - 1) == 0)

static void PCBuildPrivate(CdamPC* pc, void* A, void* config) {
	CdamParMat* mat = (CdamParMat*)A;
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
		CdamParMatGetDiag(mat, pc->diag, bs);
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
			dgetriBatched(bs, input_batch, bs, pivot, output_batch, bs, info, num_node);
			CdamMemcpy(pc->diag, inv, num_node * bs * bs * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);


			CdamFree(inv, num_node * bs * bs * sizeof(value_type), DEVICE_MEM);
			CdamFree(input_batch, num_node * sizeof(value_type*) * 2, DEVICE_MEM);
			CdamFree(info, num_node * sizeof(int) * (bs + 1), DEVICE_MEM);
			CdamFree(h_batch, num_node * sizeof(value_type*), HOST_MEM);
		}
	}
	else if(streq_c(JSONGetItem(json, "Type")->valuestring, "Schur")) {
		pc->type = PC_TYPE_SCHUR;
		if(streq_c(JSONGetItem(json, "SchurType")->valuestring, "Full")) {
			pc->schur_type = PC_SCHUR_FULL;
		}
		else if(streq_c(JSONGetItem(json, "SchurType")->valuestring, "Diag")) {
			pc->schur_type = PC_SCHUR_DIAG;
		}
		else if(streq_c(JSONGetItem(json, "SchurType")->valuestring, "Upper")) {
			pc->schur_type = PC_SCHUR_UPPER;
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

static void PCSchurGenAp(CdamPC* pc) {

}

static void PCSchurGenS(CdamPC* pc) {

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
													0.0, y + displ, 1, bs,
													count / bs);
		}
	}
	else if(pc->type == PC_TYPE_SCHUR) {
		/* Generate Ap */
		PCSchurGenAp(pc);

		/* Generate S */
		PCSchurGenS(pc);

		if(pc->schur_type & PC_SCHUR_LOWER) {
			/* Solve [I O; -C*inv{A} I] [y0, y1] = [x0, x1]*/
		}
		
		/* Solve [A O; O S] [y0, y1] = [x0, x1] */
		/* TODO: tmp[displ:displ+count] *= inv{Ap} */
		/* TODO: Implement Schur complement */
		/* TODO: tmp[displ1+displ1+count1] *= inv{S} */

		if(pc->schur_type & PC_SCHUR_UPPER) {
			/* Solve [I O; -C*inv{A} I] [y0, y1] = [x0, x1]*/
		}
	}
	else if(pc->type == PC_TYPE_KSP) {
		CdamKrylovSolve((CdamKrylov*)pc->ksp, pc->mat, y + displ, x + displ);
	}
	else if(pc->type == PC_TYPE_CUSTOM) {
		pc->op->apply(pc, x, y);
	}

	if(pc->next) {
		PCApplyPrivate(pc->next, x, y);
	}
}

static void PCRichardsonAllocPrivate(CdamPC* pc, void* A, void* config) {
	UNUSED(pc);
	UNUSED(A);
	UNUSED(config);
}

static void PCRichardsonSetupPrivate(CdamPC* pc, void* A, void* config) {
	cJSON* json = (cJSON*)config;
	pc->omega = JSONGetItem(json, "omega")->valuedouble;
	UNUSED(A);
}

static void PCRichardsonDestroyPrivate(CdamPC* pc) {
	UNUSED(pc);
}

static void PCRicahrdsonApplyPrivate(CdamPC* pc, value_type* x, value_type* y) {
	index_type n = pc->n_vert;
	index_type displ = pc->displ;
	index_type count = pc->count;
	value_type omega = pc->omega;
	CdamMemcpy(y + displ * n, x + displ * n, n* count * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
	dscal(n * count, omega, y + displ * n, 1);
}

static void PCJacobiAllocPrivate(CdamPC* pc, void* A, void* config) {
	cJSON* json = (cJSON*)config;
	UNUSED(json);
	pc->mat = A;
	pc->bs = JSONGetItem(json, "bs")->valueint;
	pc->n_vert = CdamParMatNumRowAll(A) / pc->bs;
	pc->diag = CdamTMalloc(value_type, pc->n_vert * pc->bs * pc->bs, DEVICE_MEM);
}

static void PCJacobiSetupPrivate(CdamPC* pc, void* A, void* config) {
	cJSON* json = (cJSON*)config;
	UNUSED(json);
	index_type bs = pc->bs;
	index_type n = CdamParMatNumRowAll(A) / bs;
	value_type* pc_ctx = (value_type*)pc->diag;
	CdamParMatGetDiag(A, pc_ctx, bs);
	value_type *inv, **input_batch, **output_batch, **h_batch;
	int* info, *pivot;

	if(bs == 1) {
		VecPointwiseInv(pc_ctx, n);
	}
	else if(bs > 1) {
		inv = CdamTMalloc(value_type, n * bs * bs, DEVICE_MEM);
		input_batch = CdamTMalloc(value_type*, n * 2, DEVICE_MEM);
		output_batch = input_batch + n;
		info = CdamTMalloc(int, n * (bs + 1), DEVICE_MEM);
		pivot = info + n;

		h_batch = CdamTMalloc(value_type*, n, HOST_MEM);
		for(int i = 0; i < n; i++) {
			h_batch[i] = pc_ctx + i * bs * bs;
		}
		CdamMemcpy(input_batch, h_batch, n * sizeof(value_type*), DEVICE_MEM, HOST_MEM);
		for(int i = 0; i < n; i++) {
			h_batch[i] = inv + i * bs * bs;
		}
		CdamMemcpy(output_batch, h_batch, n * sizeof(value_type*), DEVICE_MEM, HOST_MEM);
		dgetrfBatched(bs, input_batch, bs, pivot, info, n);
		dgetriBatched(bs, input_batch, bs, pivot, output_batch, bs, info, n);
		CdamMemcpy(pc_ctx, inv, n * bs * bs * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);

		CdamFree(inv, n * bs * bs * sizeof(value_type), DEVICE_MEM);
		CdamFree(input_batch, n * sizeof(value_type*) * 2, DEVICE_MEM);
		CdamFree(info, n * sizeof(int) * (bs + 1), DEVICE_MEM);
		CdamFree(h_batch, n * sizeof(value_type*), HOST_MEM);
	}
}

static void PCJacobiDestroyPrivate(CdamPC* pc) {
	CdamFree(pc->diag, pc->n_vert * pc->bs * pc->bs * sizeof(value_type), DEVICE_MEM);
}

static void PCJacobiApplyPrivate(CdamPC* pc, value_type* x, value_type* y) {
	index_type bs = pc->bs;
	value_type* diag = (value_type*)pc->diag;
	
	if(bs == 1) {
		VecPointwiseMult(x, diag, y, pc->n_vert);
	}
	else if(bs > 1) {
		dgemvStridedBatched(BLAS_N, bs, bs, 1.0,
												diag, bs, bs * bs,
												x, 1, bs, 0.0,
												y, 1, bs,
												pc->n_vert);
	}
}

static void PCSchurAllocPrivate(CdamPC* pc, void* A, void* config) {
	/* TODO */
}
static void PCSchurSetupPrivate(CdamPC* pc, void* A, void* config) {
	/* TODO */
}
static void PCSchurDestroyPrivate(CdamPC* pc) {
	/* TODO */
}
static void PCSchurApplyPrivate(CdamPC* pc, value_type* x, value_type* y) {
	/* TODO */
}

static void PCKSPAllocPrivate(CdamPC* pc, void* A, void* config) {
	UNUSED(A);
	UNUSED(config);
	CdamKrylovCreate((CdamKrylov**)&pc->ksp);
}
static void PCKSPSetupPrivate(CdamPC* pc, void* A, void* config) {
	CdamKrylovSetup((CdamKrylov*)pc->ksp, A, config);
}
static void PCKSPDestroyPrivate(CdamPC* pc) {
	CdamKrylovDestroy((CdamKrylov*)pc->ksp);
}
static void PCKSPApplyPrivate(CdamPC* pc, value_type* x, value_type* y) {
	CdamKrylovSolve((CdamKrylov*)pc->ksp, pc->mat, y, x);
}

static void PCCompositeAllocPrivate(CdamPC* pc, void* A, void* config) {
	UNUSED(A);
	UNUSED(config);
}


__END_DECLS__

