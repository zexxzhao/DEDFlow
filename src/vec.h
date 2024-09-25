#ifndef __VEC_H__
#define __VEC_H__

#include <mpi.h>
#include <strings.h>
#include "common.h"

__BEGIN_DECLS__

#define CDAM_VEC_MAX_NUM_COMPONENTS 16
typedef struct CdamLayout CdamLayout;
struct CdamLayout {
	index_type num[2];
	index_type num_components;
	index_type component_offsets[CDAM_VEC_MAX_NUM_COMPONENTS];
};

#define CdamLayoutNumOwned(layout) ((layout)->num[0])
#define CdamLayoutNumGhosted(layout) ((layout)->num[1])
#define CdamLayoutNumComponents(layout) ((layout)->num_components)
#define CdamLayoutComponentOffset(layout) ((layout)->component_offsets)
#define CdamLayoutLen(layout) (\
		{ \
			index_type len = CdamLayoutNumOwned(layout) + CdamLayoutNumGhost(layout); \
			index_type num_components = CdamLayoutNumComponents(layout); \
			len *= CdamLayoutComponentOffset(layout)[num_components]; \
			len; \
		})

void CdamLayoutCreate(CdamLayout** layout, void* config);
void CdamLayoutDestroy(CdamLayout* layout);
void CdamLayoutSetup(CdamLayout* layout, void* mesh);

void VecDot(void* x, void* y, void* layout, void* result, void* results);
void VecNorm(void* vec, void* layout, void* result, void* results);

__END_DECLS__
#endif /* __VEC_H__ */
