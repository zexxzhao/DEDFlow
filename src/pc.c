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

static void RestrictVecValueType(value_type* y, value_type* x, index_type count, index_type* index) {
	RestrictVec(y, x, count, index, sizeof(value_type));
}

static void ProlongateVecValueType(value_type* y, value_type* x, index_type count, index_type* index) {
	ProlongateVec(y, x, count, index, sizeof(value_type));
}

static void PCAllocPrivate(CdamPC* pc, void* A, void* config);
static void PCSetupPrivate(CdamPC* pc, void* config);
static void PCApplyPrivate(CdamPC* pc, value_type* x, value_type* y);


static void PCRichardsonAllocPrivate(CdamPC* pc, void* A, void* config) {
	UNUSED(pc);
	UNUSED(A);
	UNUSED(config);
}

static void PCRichardsonSetupPrivate(CdamPC* pc, void* config) {
	cJSON* json = (cJSON*)config;
	pc->omega = JSONGetItem(json, "omega")->valuedouble;
}

static void PCRichardsonDestroyPrivate(CdamPC* pc) {
	UNUSED(pc);
}

static void PCRicahrdsonApplyPrivate(CdamPC* pc, value_type* x, value_type* y) {
	index_type count = pc->count;
	value_type omega = pc->omega;
	CdamMemcpy(y, x, count * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
	dscal(count, omega, y, 1);
}



static void PCJacobiAllocPrivate(CdamPC* pc, void* A, void* config) {
	index_type n;
	cJSON* json = (cJSON*)config;
	UNUSED(json);
	pc->mat = A;
	pc->bs = JSONGetItem(json, "bs")->valueint;
	n = CdamParMatNumRowAll(A) / pc->bs;
	pc->diag = CdamTMalloc(value_type, n * pc->bs * pc->bs, DEVICE_MEM);
	if(pc->bs > 1) {
		pc->diag_inv = CdamTMalloc(value_type, n * pc->bs * pc->bs, DEVICE_MEM);
		pc->diag_batch = CdamTMalloc(value_type*, n * 2, DEVICE_MEM);
		pc->diag_ipiv = CdamTMalloc(int, n * (pc->bs + 1), DEVICE_MEM);

		value_type** h_batch = CdamTMalloc(value_type*, n, HOST_MEM);
		for(int i = 0; i < n; i++) {
			h_batch[i] = pc->diag + i * pc->bs * pc->bs;
		}
		CdamMemcpy(pc->diag_batch, h_batch, n * sizeof(value_type*), DEVICE_MEM, HOST_MEM);
		for(int i = 0; i < n; i++) {
			h_batch[i] = pc->diag_inv + i * pc->bs * pc->bs;
		}
		CdamMemcpy(pc->diag_batch + n, h_batch, n * sizeof(value_type*), DEVICE_MEM, HOST_MEM);
		CdamFree(h_batch, n * sizeof(value_type*), HOST_MEM);
	}
}

static void PCJacobiSetupPrivate(CdamPC* pc, void* config) {
	cJSON* json = (cJSON*)config;
	UNUSED(json);
	index_type bs = pc->bs;
	index_type n = CdamParMatNumRowAll(pc->mat) / bs;
	value_type* diag = (value_type*)pc->diag;
	CdamParMatGetDiag(pc->mat, diag, bs);
	value_type **input_batch, **output_batch;
	int* info, *pivot;

	if(bs == 1) {
		VecPointwiseInv(diag, n);
	}
	else if(bs > 1) {
		input_batch = pc->diag_batch;
		output_batch = input_batch + n;
		info = pc->diag_ipiv;
		pivot = info + n;

		/* Blockwise LU decomposition */
		dgetrfBatched(bs, input_batch, bs, pivot, info, n);
		/* Blockwise Inverse */
		dgetriBatched(bs, input_batch, bs, pivot, output_batch, bs, info, n);
	}
}

static void PCJacobiDestroyPrivate(CdamPC* pc) {
	index_type bs = pc->bs;
	index_type count = pc->count;
	if(bs == 1) {
		CdamFree(pc->diag, count * sizeof(value_type), DEVICE_MEM);
	}
	else if(bs > 1) {
		CdamFree(pc->diag, count * bs * sizeof(value_type), DEVICE_MEM);
		CdamFree(pc->diag_inv, count * bs * sizeof(value_type), DEVICE_MEM);
		CdamFree(pc->diag_batch, count * sizeof(value_type*) * 2, DEVICE_MEM);
		CdamFree(pc->diag_ipiv, count * sizeof(int) * (bs + 1), DEVICE_MEM);
	}
}

static void PCJacobiApplyPrivate(CdamPC* pc, value_type* x, value_type* y) {
	index_type bs = pc->bs;
	value_type* diag = (value_type*)pc->diag;
	value_type* diag_inv = (value_type*)pc->diag_inv;
	
	if(bs == 1) {
		VecPointwiseMult(x, diag, y, pc->count);
	}
	else if(bs > 1) {
		dgemvStridedBatched(BLAS_N, bs, bs, 1.0,
												diag_inv, bs, bs * bs,
												x, 1, bs, 0.0,
												y, 1, bs,
												pc->count / bs);
	}
}

static void PCSchurAllocPrivate(CdamPC* pc, void* A, void* config) {
	cJSON* json = (cJSON*)config;
	if(strcmp(JSONGetItem(json, "SchurType")->valuestring, "Full") == 0) {
		pc->schur_type = PC_SCHUR_FULL;
	}
	else if(strcmp(JSONGetItem(json, "SchurType")->valuestring, "Diag") == 0) {
		pc->schur_type = PC_SCHUR_DIAG;
	}
	else if(strcmp(JSONGetItem(json, "SchurType")->valuestring, "Upper") == 0) {
		pc->schur_type = PC_SCHUR_UPPER;
	}
	else {
		fprintf(stderr, "Unknown SchurType\n");
		exit(1);
	}

	/* Allocate memory for Schur complement */
	CdamParMatCreate(MPI_COMM_WORLD, &pc->schur_mat[0][0]);
	CdamParMatCreate(MPI_COMM_WORLD, &pc->schur_mat[0][1]);
	CdamParMatCreate(MPI_COMM_WORLD, &pc->schur_mat[1][0]);
	CdamParMatCreate(MPI_COMM_WORLD, &pc->schur_mat[1][1]);
	
	/* Allocate memory for SubPC of A[0][0] */
	pc->schur_pc[0] = CdamTMalloc(CdamPC, 1, HOST_MEM);
	PCAllocPrivate(pc->schur_pc[0], A, JSONGetItem(json, "SubPC0"));
	/* Allocate memory for SubPC of A[1][1],
	 * which will be used for Schur complement S=A[1][1] - A[1][0] * inv(A[0][0]) * A[0][1] */
	pc->schur_pc[1] = CdamTMalloc(CdamPC, 1, HOST_MEM);
	PCAllocPrivate(pc->schur_pc[1], A, JSONGetItem(json, "SubPC1"));
}

static void PCSchurSetupPrivate(CdamPC* pc, void* config) {
	index_type n = CdamParMatNumRowAll(pc->mat);
	index_type count = pc->count, *index = pc->index;
	CdamParMatGetSubmat(pc->mat, count, index, count, index, pc->schur_mat[0][0]);
	CdamParMatGetSubmat(pc->mat, count, index, n - count, index + count, pc->schur_mat[0][1]);
	CdamParMatGetSubmat(pc->mat, n - count, index + count, count, index, pc->schur_mat[1][0]);
	CdamParMatGetSubmat(pc->mat, n - count, index + count, n - count, index + count, pc->schur_mat[1][1]);

	pc->schur_pc[0]->mat = pc->schur_mat[0][0];
	pc->schur_pc[1]->mat = pc->schur_mat[1][1];

	PCSetupPrivate(pc->schur_pc[0], config);
	PCSetupPrivate(pc->schur_pc[1], config);

}
static void PCSchurDestroyPrivate(CdamPC* pc) {
	CdamParMatDestroy(&pc->schur_mat[0][0]);
	CdamParMatDestroy(&pc->schur_mat[0][1]);
	CdamParMatDestroy(&pc->schur_mat[1][0]);
	CdamParMatDestroy(&pc->schur_mat[1][1]);
	CdamPCDestroy(pc->schur_pc[0]);
	CdamPCDestroy(pc->schur_pc[1]);
}

static void PCSchurApplyPrivate(CdamPC* pc, value_type* x, value_type* y) {

	index_type n = CdamParMatNumRowAll(pc->mat);
	index_type count = pc->count;
	value_type* xtmp = pc->buff;
	value_type* ytmp = xtmp + n;

	/* Restrict x to xtmp */
	RestrictVecValueType(xtmp, x, pc->count, pc->index);
	RestrictVecValueType(xtmp + count, x, n - count, pc->index + count);

	/* Solve [I O; -C*inv{A} I] [y0, y1] = [x0, x1]*/
	if(pc->schur_type & PC_SCHUR_LOWER) {
		CdamMemcpy(ytmp, xtmp, pc->count * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);	
		PCApplyPrivate(pc->schur_pc[0], xtmp, ytmp);
		CdamMemcpy(ytmp + count, xtmp, (n - count) * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
		CdamParMatMultAdd(-1.0, pc->schur_mat[0][1], xtmp, 1.0, ytmp + count);
	}
	
	/* Solve [A O; O S] [y0, y1] = [x0, x1] */
	CdamMemcpy(xtmp, ytmp, n * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
	CdamMemset(ytmp, 0, n * sizeof(value_type), DEVICE_MEM);
	PCApplyPrivate(pc->schur_pc[0], xtmp, ytmp);
	PCApplyPrivate(pc->schur_pc[1], xtmp + count, ytmp + count);

	/* Solve [I O; -C*inv{A} I] [y0, y1] = [x0, x1]*/
	CdamMemcpy(xtmp, ytmp, n * sizeof(value_type), DEVICE_MEM, DEVICE_MEM);
	if(pc->schur_type & PC_SCHUR_UPPER) {
		PCApplyPrivate(pc->schur_pc[1], xtmp + count, ytmp + count);
		CdamParMatMultAdd(-1.0, pc->schur_mat[1][0], xtmp + count, 1.0, ytmp);
	}

	/* Prolongate ytmp to y */
	ProlongateVecValueType(y, ytmp, pc->count, pc->index);
	ProlongateVecValueType(y, ytmp + count, n - count, pc->index + count);

}

static void PCKSPAllocPrivate(CdamPC* pc, void* A, void* config) {
	UNUSED(A);
	UNUSED(config);
	CdamKrylovCreate((CdamKrylov**)&pc->ksp);
}
static void PCKSPSetupPrivate(CdamPC* pc, void* config) {
	CdamKrylovSetup((CdamKrylov*)pc->ksp, pc->mat, config);
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
	cJSON* json = (cJSON*)config;

	if(strcmp(JSONGetItem(json, "Type")->valuestring, "Additive") == 0) {
		pc->comp_type = PC_COMPOSITE_ADDITIVE;
	}
	else if(strcmp(JSONGetItem(json, "Type")->valuestring, "Multiplicative") == 0) {
		pc->comp_type = PC_COMPOSITE_MULTIPLICATIVE;
	}
	else {
		fprintf(stderr, "Unknown PC type\n");
		exit(1);
	}

	pc->child = CdamTMalloc(CdamPC, 2, HOST_MEM);

	PCAllocPrivate(pc->child + 0, A, JSONGetItem(json, "SubPC0"));
	PCAllocPrivate(pc->child + 1, A, JSONGetItem(json, "SubPC1"));
}

static void PCCompositeSetupPrivate(CdamPC* pc, void* config) {
	PCSetupPrivate(pc->child + 0, config);
	PCSetupPrivate(pc->child + 1, config);
}

static void PCCompositeApplyPrivate(CdamPC* pc, value_type* x, value_type* y) {
	if(pc->comp_type == PC_COMPOSITE_ADDITIVE) {
		PCApplyPrivate(pc->child + 0, x, y);
		PCApplyPrivate(pc->child + 1, x, y);
	}
	else if(pc->comp_type == PC_COMPOSITE_MULTIPLICATIVE) {
		value_type* tmp = pc->buff;
		PCApplyPrivate(pc->child + 0, x, tmp);
		PCApplyPrivate(pc->child + 1, tmp, y);
	}
}

static void PCAllocPrivate(CdamPC* pc, void* A, void* config) {
	if(pc == NULL) {
		return;
	}

	pc->op->alloc(pc, A, config);
}

static void PCSetupPrivate(CdamPC* pc, void* config) {
	if(pc == NULL) {
		return;
	}

	pc->op->setup(pc, config);
}


static void PCApplyPrivate(CdamPC* pc, value_type* x, value_type* y) {
	if(pc == NULL) {
		return;
	}

	PCApplyPrivate(pc->next, x, y);
	PCApplyPrivate(pc, x, y);
}


void CdamPCCreate(void* mat, CdamPC** pc) {
	*pc = CdamTMalloc(CdamPC, 1, HOST_MEM);
	CdamMemset(*pc, 0, sizeof(CdamPC), HOST_MEM);
	(*pc)->mat = mat;

}

void CdamPCSetup(CdamPC* pc, void* config) {
	if(pc == NULL) {
		return;
	}

	pc->op->setup(pc, config);
}

void CdamPCApply(CdamPC* pc, value_type* x, value_type* y) {
	if(pc == NULL) {
		return;
	}

	PCApplyPrivate(pc, x, y);
}

void CdamPCDestroy(CdamPC* pc) {
	if(pc == NULL) {
		return;
	}

	pc->op->destroy(pc);
	CdamFree(pc, 1, HOST_MEM);
}

__END_DECLS__

