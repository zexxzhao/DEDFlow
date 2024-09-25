#ifndef __MATRIX_UTIL_H__
#define __MATRIX_UTIL_H__

#include "common.h"

__BEGIN_DECLS__

typedef enum {
	MAT_TYPE_NONE = 0,
	MAT_TYPE_DENSE = 1, /* Dense matrix */
	MAT_TYPE_CSR = 2,   /* Compressed Sparse Row matrix */
	MAT_TYPE_SELL = 3,   /* Slice ELL matrix */
	MAT_TYPE_VIRTUAL = 4,    /* Virtual matrix */
	MAT_TYPE_CUSTOM = 5 /* Custom matrix */
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
	MAT_ASSEMBLED = 0,
	MAT_DISASSEMBLED = 1
} MatAssemblyType;

__END_DECLS__
#endif /* __MATRIX_UTIL_H__ */
