#ifndef __VEC_H__
#define __VEC_H__

#include <mpi.h>
#include "common.h"

__BEGIN_DECLS__

enum CDAM_VecType {
	CDAM_VEC_TYPE_NONE = 0x0,
	CDAM_VEC_TYPE_HYPRE = 0x1
};
typedef enum CDAM_VecType CDAM_VecType;

typedef struct hypre_IJVector_struct hypre_IJVector;
typedef struct hypre_IJVector_struct* HYPRE_IJVector;
typedef struct hypre_ParVector_struct hypre_ParVector;
typedef struct hypre_ParVector_struct* HYPRE_ParVector;
typedef struct CDAM_Vec CDAM_Vec;
struct CDAM_Vec {
	MPI_Comm comm;
	index_type partition[2];
	index_type num_components;
	index_type* bs;
	HYPRE_IJVector* vecs;

};

#define CDAM_VecComm(vec) ((vec)->comm)
#define CDAM_VecPartitionLower(vec) ((vec)->partition[0])
#define CDAM_VecPartitionUpper(vec) ((vec)->partition[1])
#define CDAM_VecNumComponents(vec) ((vec)->num_components)
#define CDAM_VecComponentSize(vec) ((vec)->bs)
#define CDAM_VecSub(vec) ((vec)->vecs)
#define CDAM_VecSubSeqVec(vec, i) \
	(((HYPRE_ParVector)(CDAM_VecSub(vec)[i])->object)->local_vector)
#define CDAM_VecRawPtr(vec, i) \
	(CDAM_VecSubSeqVec(vec, i)->data)

void CDAM_VecCreate(MPI_Comm comm, void* mesh, CDAM_Vec** vec);
void CDAM_VecSetNumComponents(CDAM_Vec* vec, index_type num_components);
void CDAM_VecSetComponentSize(CDAM_Vec* vec, index_type component, index_type size);
void CDAM_VecInitialize(CDAM_Vec* vec);
void CDAM_VecDestroy(CDAM_Vec* vec);

/* void CDAM_VecSetValues(CDAM_Vec* vec, index_type n, index_type indices[], value_type values[]); */
/* void CDAM_VecGetValues(CDAM_Vec* vec, index_type n, index_type indices[], value_type values[]); */

void CDAM_VecGetArray(CDAM_Vec* vec, value_type** array);
void CDAM_VecRestoreArray(CDAM_Vec* vec, value_type** array);

void CDAM_VecAssemble(CDAM_Vec* vec);

void CDAM_VecCopy(CDAM_Vec* x, CDAM_Vec* y);
void CDAM_VecAXPY(value_type alpha, CDAM_Vec* x, CDAM_Vec* y);
void CDAM_VecScale(CDAM_Vec* vec, value_type alpha);
void CDAM_VecZero(CDAM_Vec* vec);

void CDAM_VecInnerProd(CDAM_Vec* x, CDAM_Vec* y, value_type* result);
void CDAM_VecNorm(CDAM_Vec* vec, value_type* result);

void CDAM_VecView(CDAM_Vec* vec, FILE* ostream);

#define WrapVec(IJVec) ((CDAM_Vec){ \
	.comm = hypre_IJVectorComm(IJVec), \
	.partition = {hypre_IJVectorPartitioning(IJVec)[0], hypre_IJVectorPartitioning(IJVec)[1]}, \
	.num_components = 1, \
	.bs = {1}, \
	.vecs = &IJVec \
	})

__END_DECLS__
#endif /* __VEC_H__ */
