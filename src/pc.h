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
	PC_CUSTOM = 0x8
};

typedef enum CdamPCType CdamPCType;

enum CdamPCDecomposeType {
	PC_DECOMPOSE_ADDITIVE = 0x0,
	PC_DECOMPOSE_MULTIPLICATIVE = 0x1,
	PC_DECOMPOSE_SCHUR_FULL = 0x2,
	PC_DECOMPOSE_SCHUR_DIAG = 0x3,
	PC_DECOMPOSE_SCHUR_UPPER = 0x4
};

typedef enum CdamPCDecomposeType CdamPCDecomposeType;

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

	CdamPC* next;
	CdamPC* prev;
	CdamPC* child;

	CdamPCOps op[1];
	CdamPCType type;

	void* ksp;
	void* mat;

	index_type n;

	/* Richardson */
	value_type omega;

	/* Jacobi */
	index_type bs;
	value_type* diag;

	/* Decomposition */
	index_type n_sec;
	index_type* offset;

	/* Custom */
	void* ctx;
};

struct CdamPC {
	CdamPCType type;
	void* mat;
	CdamPCOps op[1];
	void* data;
};

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

CdamPC* CdamPCCreateNone(Matrix* mat, index_type n);
CdamPC* CdamPCCreateJacobi(Matrix* mat, index_type bs, void* cublas_handle);
CdamPC* CdamPCCreateDecomposition(Matrix* mat, index_type n, const index_type* offset, void* cublas_handle);
CdamPC* CdamPCCreateAMGX(Matrix* mat, void* options);
void CdamPCSetup(CdamPC* pc, void* config);
void CdamPCDestroy(CdamPC* pc);
void CdamPCApply(CdamPC* pc, f64* x, f64* y);


__END_DECLS__
#endif /* __PC_H__ */
