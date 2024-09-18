#ifndef __VEC_H__
#define __VEC_H__

#include <mpi.h>
#include <strings.h>
#include "common.h"

__BEGIN_DECLS__

#define CDAM_VEC_MAX_NUM_COMPONENTS 64

enum CdamVecType {
	CdamVEC_TYPE_NONE = 0x0,
	CdamVEC_TYPE_HYPRE = 0x1
};
typedef enum CdamVecType CdamVecType;

typedef struct CdamVec CdamVec;
struct CdamVec {
	MPI_Comm comm;
	index_type partition[3];
	index_type num_components;
	index_type bs[CDAM_VEC_MAX_NUM_COMPONENTS];
	
};

#define CdamVecComm(vec) ((vec)->comm)
#define CdamVecPartitionLower(vec) ((vec)->partition[0])
#define CdamVecPartitionUpper(vec) ((vec)->partition[1])
#define CdamVecPartitionCapacity(vec) ((vec)->partition[2])
#define CdamVecLen(vec) ({\
	index_type len = 0; \
	for (index_type i = 0; i < (vec)->num_components; i++) { \
		len += (vec)->bs[i] * CdamVecPartitionCapacity(vec); \
	} \
	len; \
})
#define CdamVecNumComponents(vec) ((vec)->num_components)
#define CdamVecComponentSize(vec) ((vec)->bs)
#define CdamVecSub(vec) ((vec)->vecs)
#define CdamVecSubSeqVec(vec, i) \
	(((HYPRE_ParVector)(CdamVecSub(vec)[i])->object)->local_vector)
#define CdamVecRawPtr(vec, i) \
	(CdamVecSubSeqVec(vec, i)->data)

void CdamVecCreate(MPI_Comm comm, void* mesh, CdamVec** vec);
void CdamVecSetNumComponents(CdamVec* vec, index_type num_components);
void CdamVecSetComponentSize(CdamVec* vec, index_type component, index_type size);
void CdamVecInitialize(CdamVec* vec);
void CdamVecDestroy(CdamVec* vec);

/* void CdamVecSetValues(CdamVec* vec, index_type n, index_type indices[], value_type values[]); */
/* void CdamVecGetValues(CdamVec* vec, index_type n, index_type indices[], value_type values[]); */

void CdamVecGetArray(CdamVec* vec, value_type** array);
void CdamVecRestoreArray(CdamVec* vec, value_type** array);

void CdamVecAssemble(CdamVec* vec);

void CdamVecCopy(CdamVec* x, CdamVec* y);
void CdamVecAXPY(value_type alpha, CdamVec* x, CdamVec* y);
void CdamVecScale(CdamVec* vec, value_type alpha);
void CdamVecZero(CdamVec* vec);


void CdamVecInnerProd(CdamVec* x, CdamVec* y, value_type* result, value_type* separated_result);
void CdamVecNorm(CdamVec* vec, value_type* result, value_type* separated_result);

void CdamVecView(CdamVec* vec, FILE* ostream);


#define WrapVec(IJVec) ((CdamVec){ \
	.comm = hypre_IJVectorComm(IJVec), \
	.partition = {hypre_IJVectorPartitioning(IJVec)[0], hypre_IJVectorPartitioning(IJVec)[1]}, \
	.num_components = 1, \
	.bs = {1}, \
	.vecs = IJVec \
	})

typedef struct CdamVecLayout CdamVecLayout;
struct CdamVecLayout {
	index_type num[2];
	index_type num_components;
	index_type component_offsets[CDAM_VEC_MAX_NUM_COMPONENTS];
};

#define CdamVecLayoutNumOwned(layout) ((layout)->num[0])
#define CdamVecLayoutNumGhost(layout) ((layout)->num[1])
#define CdamVecLayoutNumComponents(layout) ((layout)->num_components)
#define CdamVecLayoutComponentOffset(layout) ((layout)->component_offsets)
#define CdamVecLayoutLen(layout) (\
		{ \
			index_type len = CdamVecLayoutNumOwned(layout) + CdamVecLayoutNumGhost(layout); \
			index_type num_components = CdamVecLayoutNumComponents(layout); \
			len *= CdamVecLayoutComponentOffset(layout)[num_components]; \
			len; \
		})

void CdamVecLayoutCreate(CdamVecLayout** layout, void* config);
void CdamVecLayoutDestroy(CdamVecLayout* layout);
void CdamVecLayoutSetup(CdamVecLayout* layout, void* mesh);

void VecDot(void* x, void* y, void* layout, void* result, void* results);
void VecNorm(void* vec, void* layout, void* result, void* results);

__END_DECLS__
#endif /* __VEC_H__ */
