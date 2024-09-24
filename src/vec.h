#ifndef __VEC_H__
#define __VEC_H__

#include <mpi.h>
#include <strings.h>
#include "common.h"

__BEGIN_DECLS__

#define CDAM_VEC_MAX_NUM_COMPONENTS 16
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
