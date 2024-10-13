#ifndef __PC_H__
#define __PC_H__
#ifdef USE_AMGX
#include <amgx_c.h>
#endif
#include "common.h"

__BEGIN_DECLS__

enum CdamPCType {
	PC_TYPE_NONE = 0x0,
	PC_TYPE_RICHARDSON = 0x1,
	PC_TYPE_JACOBI = 0x2,
	PC_TYPE_SCHUR = 0x4,
	PC_TYPE_KSP = 0x8,
	PC_TYPE_COMPOSITE = 0x10,
	PC_TYPE_CUSTOM = 0xff
};

typedef enum CdamPCType CdamPCType;

enum CdamPCSchurType {
	PC_SCHUR_DIAG = 0x0,
	PC_SCHUR_UPPER = 0x1,
	PC_SCHUR_LOWER = 0x2,
	PC_SCHUR_FULL = PC_SCHUR_UPPER | PC_SCHUR_LOWER,
};

typedef enum CdamPCSchurType CdamPCSchurType;

enum CdamPCCompositeType {
	PC_COMPOSITE_ADDITIVE = 0x1,
	PC_COMPOSITE_MULTIPLICATIVE = 0x2
};

typedef enum CdamPCCompositeType CdamPCCompositeType;

struct CdamPCOps {
	void (*alloc)(void*, void*, void*);
	void (*setup)(void*, void*);
	void (*destroy)(void*);
	void (*apply)(void*, value_type*, value_type*);
};
typedef struct CdamPCOps CdamPCOps;

struct CdamPC {

	struct CdamPC* next;

	CdamPCType type;

	void* mat;

	index_type count;
	index_type* index;
	value_type* buff;

	CdamPCOps op[1];

	/* Richardson */
	value_type omega;

	/* Jacobi */
	index_type bs;
	value_type* diag;
	value_type* diag_inv;
	value_type** diag_batch;
	int* diag_ipiv;

	/* Schur */
	CdamPCSchurType schur_type;
	void* schur_mat[2][2];
	struct CdamPC* schur_pc[2];

	/* KSP */
	void* ksp;

	/* Composite */
	CdamPCCompositeType comp_type;
	struct CdamPC* child;

	/* Custom */
	void* ctx;
};

typedef struct CdamPC CdamPC;

// struct CdamPC {
// 	CdamPCType type;
// 	void* mat;
// 	CdamPCOps op[1];
// 	void* data;
// };

void CdamPCCreate(void *mat, CdamPC** pc);
void CdamPCSetup(CdamPC* pc, void* config);
void CdamPCDestroy(CdamPC* pc);
void CdamPCApply(CdamPC* pc, f64* x, f64* y);

/* This function is used to set the subdomain for the Schur complement */
void CdamPCSetSubdomain(CdamPC* pc, index_type n, index_type* index);


__END_DECLS__
#endif /* __PC_H__ */
