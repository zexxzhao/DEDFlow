#ifndef __MATRIX_UTIL_H__
#define __MATRIX_UTIL_H__

#include "common.h"

__BEGIN_DECLS__

typedef enum {
	MAT_TYPE_NONE = 0,
	MAT_TYPE_DENSE = 1, /* Dense matrix */
	MAT_TYPE_CSR = 2,   /* Compressed Sparse Row matrix */
	MAT_TYPE_SELL = 4,   /* Slice ELL matrix */
	MAT_TYPE_FS = 8,    /* Field Splitting matrix */
	MAT_TYPE_CUSTOM = 16 /* Custom matrix */
} MatType;

typedef enum {
	MAT_INITIAL = 0,
	MAT_REUSE = 1
}	MatReuse;

typedef enum {
	MAT_ROW_MAJOR = 0,
	MAT_COL_MAJOR = 1
} MatOrder;

typedef enum {
	MAT_STORAGE_COLWISE = 0, /* v0[0], ..., v0[N], v1[0], ..., v1[N], v2[0], ... */
	MAT_STORAGE_ROWWISE = 1  /* v0[0], v1[0], v2[0], ..., v0[1], v1[1], v2[1], ... */
} MatStorageMethod;

typedef enum {
	MAT_ASSEMBLED = 0,
	MAT_DISASSEMBLED = 1
} MatAssemblyType;

__END_DECLS__
#endif /* __MATRIX_UTIL_H__ */
