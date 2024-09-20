#ifndef __PC_H__
#define __PC_H__
#ifdef USE_AMGX
#include <amgx_c.h>
#endif
#include "common.h"
#include "matrix.h"

__BEGIN_DECLS__

/*
 * y = x / diag(A)
 */

enum CdamPCType {
	PC_NONE = 0x0,
	PC_RICHARDSON = 0x1,
	PC_JACOBI = 0x2,
	PC_SCHUR = 0x4,
	PC_KSP = 0x8,
	PC_COMPOSITE = 0x10,
	PC_CUSTOM = 0xff
};

typedef enum CdamPCType CdamPCType;

enum CdamPCSchurType {
	PC_SCHUR_DIAG = 0x0,
	PC_SCHUR_UPPER = 0x1
	PC_SCHUR_LOWER = 0x2,
	PC_SCHUR_FULL = PC_DECOMPOSITION_SCHUR_UPPER | PC_DECOMPOSITION_SCHUR_LOWER,
};

typedef enum CdamPCSchurType CdamPCSchurType;

enum CdamPCCompositeType {
	PC_COMPOSITE_ADDITIVE = 0x1,
	PC_COMPOSITE_MULTIPLICATIVE = 0x2
};

typedef enum CdamPCCompositeType CdamPCCompositeType;

typedef struct CdamPC CdamPC;
typedef struct CdamPCOps CdamPCOps;
struct CdamPCOps {
	void (*alloc)(CdamPC*, void*, void*);
	void (*setup)(CdamPC*, void*, void*);
	void (*destroy)(CdamPC*);
	void (*apply)(CdamPC*, value_type*, value_type*);
};


struct CdamPC {

	CdamPC* next;

	CdamPCType type;

	void* mat;

	index_type n_vert;
	index_type displ;
	index_type count;

	value_type* buff;

	/* Richardson */
	value_type omega;

	/* Jacobi */
	index_type bs;
	value_type* diag;

	/* Schur */
	CdamPCSchurType schur_type;
	void* schur_compl;
	index_type displ1;
	index_type count1;

	/* KSP */
	void* ksp;

	/* Composite */
	CdamPCCompositeType comp_type;
	CdamPC* child;

	/* Custom */
	CdamPCOps op[1];
	void* ctx;
};

// struct CdamPC {
// 	CdamPCType type;
// 	void* mat;
// 	CdamPCOps op[1];
// 	void* data;
// };

CdamPC* CdamPCCreateNone(Matrix* mat, index_type n);
CdamPC* CdamPCCreateJacobi(Matrix* mat, index_type bs, void* cublas_handle);
CdamPC* CdamPCCreateDecomposition(Matrix* mat, index_type n, const index_type* offset, void* cublas_handle);
CdamPC* CdamPCCreateAMGX(Matrix* mat, void* options);
void CdamPCSetup(CdamPC* pc, void* config);
void CdamPCDestroy(CdamPC* pc);
void CdamPCApply(CdamPC* pc, f64* x, f64* y);


__END_DECLS__
#endif /* __PC_H__ */
