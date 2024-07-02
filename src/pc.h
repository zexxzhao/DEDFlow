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

enum PCType {
	PC_NONE = 0x0,
	PC_JACOBI = 0x1,
	PC_DECOMPOSITION = 0x2,
	PC_AMGX = 0x3,
	PC_CUSTOM = 0x4
};

typedef enum PCType PCType;

typedef struct PC PC;
typedef struct PCNone PCNone;
typedef struct PCJacobi PCJacobi;
typedef struct PCDecomposition PCDecomposition;
typedef struct PCAMGX PCAMGX;


typedef struct PCOps PCOps;
struct PCOps {
	void (*setup)(PC*);
	void (*destroy)(PC*);

	void (*apply)(PC*, value_type*, value_type*);
};


struct PC {
	PCType type;
	void* mat;
	PCOps op[1];
	void* data;
	void* cublas_handle;
};

struct PCNone {
	index_type n;
};


struct PCJacobi {
	index_type n;
	index_type bs;
	void* diag;
};

struct PCDecomposition {
	index_type n_sec;
	index_type* offset;
	PC** pc;
};

struct PCAMGX {
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

struct PCCustom {
	void* ctx;
};

PC* PCCreateNone(Matrix* mat, index_type n);
PC* PCCreateJacobi(Matrix* mat, index_type bs, void* cublas_handle);
PC* PCCreateDecomposition(Matrix* mat, index_type n, const index_type* offset, void* cublas_handle);
PC* PCCreateAMGX(Matrix* mat, void* options);
void PCSetup(PC* pc);
void PCDestroy(PC* pc);
void PCApply(PC* pc, f64* x, f64* y);


__END_DECLS__
#endif /* __PC_H__ */
