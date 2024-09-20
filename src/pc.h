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
	PC_DECOMPOSITION = 0x4,
	PC_KSP = 0x8,
	PC_COMPOSITE = 0x10,
	PC_CUSTOM = 0xff
};

typedef enum CdamPCType CdamPCType;

enum CdamPCDecompositionType {
	PC_DECOMPOSITION_SCHUR_DIAG = 0x0,
	PC_DECOMPOSITION_SCHUR_UPPER = 0x1
	PC_DECOMPOSITION_SCHUR_LOWER = 0x2,
	PC_DECOMPOSITION_SCHUR_FULL = PC_DECOMPOSITION_SCHUR_UPPER | PC_DECOMPOSITION_SCHUR_LOWER,
}	PC_DECOMPOSITION_ADDITIVE = 0x100,
	PC_DECOMPOSITION_MULTIPLICATIVE = 0x101
;

typedef enum CdamPCDecompositionType CdamPCDecompositionType;

typedef struct CdamPC CdamPC;
typedef struct CdamPCNone CdamPCNone;
typedef struct CdamPCJacobi CdamPCJacobi;
typedef struct CdamPCDecomposition CdamPCDecomposition;
typedef struct CdamPCAMGX CdamPCAMGX;


typedef struct CdamPCOps CdamPCOps;
struct CdamPCOps {
	void (*setup)(CdamPC*, void*);
	void (*destroy)(CdamPC*);

	void (*apply)(CdamPC*, value_type*, value_type*);
};


struct CdamPC {

	CdamPC* prev;
	CdamPC* next;

	CdamPCType type;

	void* mat;

	index_type n;
	index_type displ;
	index_type count;

	/* Richardson */
	value_type omega;

	/* Jacobi */
	index_type bs;
	value_type* diag;

	/* Decomposition */
	CdamPCDecompositionType dtype;
	CdamPC* child;
	value_type* tmp;
	index_type displ1;
	index_type count1;

	/* KSP */
	void* ksp;

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
