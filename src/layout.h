#ifndef __LAYOUT_H__
#define __LAYOUT_H__

#include <strings.h>
#include "common.h"

__BEGIN_DECLS__

#define CDAM_MAX_NUM_COMPONENTS (16)
struct CdamLayout {
	index_type num[4]; /* exclusive, shared, ghosted, global */
	index_type component_offsets[CDAM_MAX_NUM_COMPONENTS];
	index_type* l2g;   /* local to global */
};
typedef struct CdamLayout CdamLayout;

#define CdamLayoutNumExclusive(layout) ((layout)->num[0])
#define CdamLayoutNumShared(layout) ((layout)->num[1])
#define CdamLayoutNumGhosted(layout) ((layout)->num[2])
#define CdamLayoutNumGlobal(layout) ((layout)->num[3])
#define CdamLayoutNumOwned(layout) (CdamLayoutNumExclusive(layout) + CdamLayoutNumShared(layout))
#define CdamLayoutNodalL2G(layout) ((layout)->l2g)
#define CdamLayoutNumComponent(layout) ({\
		index_type nc = 0; \
		while((layout)->component_offsets[nc] < (layout)->component_offsets[nc + 1]) nc++; \
		nc + 1; \
		})
#define CdamLayoutComponentOffset(layout) ((layout)->component_offsets)
#define CdamLayoutLen(layout) (\
		{ \
			index_type len = CdamLayoutNumOwned(layout) + CdamLayoutNumGhosted(layout); \
			index_type num_components = CdamLayoutNumComponent(layout); \
			len *= CdamLayoutComponentOffset(layout)[num_components]; \
			len; \
		})


#define DOFMapLocal(num, o, n, c) (\
		{ \
			index_type nc = 0; \
			while((o)[nc] < (o)[nc + 1]) nc++; \
			nc++; \
			index_type dof = 0; \
			index_type node_type = (n) < (num)[0] ? 0 : \
				(n) < (num)[0] + (num)[1] ? 1 : 2; \
			dof += (node_type > 0) * (o)[nc] * (num)[0]; \
			dof += (node_type > 1) * (o)[nc] * (num)[1]; \
			index_type k = 0; \
			while(k < nc && c < (o)[k]) k++; \
			dof += (num)[n] * (o)[k] + ((o)[k + 1] - (o)[k]) * (num)[n] + c - (o)[k]; \
			dof; \
		})
#define DOFMapGlobal(num, o, n, c) ({\
			index_type ng = num[3]; \
			index_type g = CdamLayoutNodalL2G(layout)[n]; \
			index_type nc = 0; \
			while((o)[nc] < (o)[nc + 1]) nc++; \
			nc++; \
			index_type k = 0; \
			while(k < nc && c < (o)[k]) k++; \
			ng * (o)[k]	+ ((o)[k + 1] - (o)[k]) * g + c - (o)[k]; \
		})

#define CdamLayoutDOFMapLocal(layout, n, c) DOFMapLocal((layout)->num, (layout)->component_offsets, n, c)


void CdamLayoutCreate(CdamLayout** layout, void* config);
void CdamLayoutDestroy(CdamLayout* layout);
void CdamLayoutSetup(CdamLayout* layout, void* mesh);

void CdamLayoutCopyToDevice(CdamLayout* layout, CdamLayout** d_layout);

// void CdamLayoutGetNodalDOF(CdamLayout* layout, index_type node, index_type* dof);

void VecDot(void* x, void* y, void* layout, void* result, void* results);
void VecNorm(void* vec, void* layout, void* result, void* results);

__END_DECLS__
#endif /* __LAYOUT_H__ */
