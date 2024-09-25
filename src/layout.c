#include <string.h>

#include "alloc.h"
#include "blas.h"
#include "Mesh.h"
#include "json.h"
#include "layout.h"

__BEGIN_DECLS__


void CdamLayoutCreate(CdamLayout** layout, void* config) {
	*layout = CdamTMalloc(CdamLayout, 1, HOST_MEM);
	CdamMemset(*layout, 0, sizeof(CdamLayout), HOST_MEM);

	CdamLayoutNumComponents(*layout) = 4;
	for(index_type i = 0; i < CDAM_MAX_NUM_COMPONENTS + 1; i++) {
		CdamLayoutComponentOffset(*layout)[i] = 0;
	}
	CdamLayoutComponentOffset(*layout)[0] = 0;
	CdamLayoutComponentOffset(*layout)[1] = 3;
	CdamLayoutComponentOffset(*layout)[2] = 4;
	CdamLayoutComponentOffset(*layout)[3] = 5;
	CdamLayoutComponentOffset(*layout)[4] = 6;

	CdamLayoutComponentOffsetDevice(*layout) = CdamTMalloc(index_type, CDAM_MAX_NUM_COMPONENTS + 1, DEVICE_MEM);
	CdamMemcpy(CdamLayoutComponentOffsetDevice(*layout), CdamLayoutComponentOffset(*layout),
						 (CDAM_MAX_NUM_COMPONENTS + 1) * sizeof(index_type), DEVICE_MEM, HOST_MEM);

}

void CdamLayoutDestroy(CdamLayout* layout) {
	CdamFree(layout, sizeof(CdamLayout), HOST_MEM);
}

void CdamLayoutSetup(CdamLayout* layout, void* mesh) {
	CdamMesh* mesh3d = (CdamMesh*)mesh;
	index_type n_owned = CdamMeshLocalNodeEnd(mesh3d) - CdamMeshLocalNodeBegin(mesh3d);
	index_type n_shared = n_owned - mesh3d->num_exclusive_node;
	index_type n_ghost = CdamMeshNumNode(mesh3d) - n_owned;

	CdamLayoutNumExclusive(layout) = n_owned - n_shared;
	CdamLayoutNumShared(layout) = n_shared;
	CdamLayoutNumGhosted(layout) = n_ghost;

	CdamLayoutNodalL2G(layout) = CdamTMalloc(index_type, CdamMeshNumNode(mesh3d), DEVICE_MEM);
	CdamMemcpy(CdamLayoutNodalL2G(layout), mesh3d->nodal_map_l2g_interior,
						 CdamMeshNumNode(mesh3d) * sizeof(index_type), DEVICE_MEM, DEVICE_MEM);

}

void CdamLayoutCopyToDevice(CdamLayout* layout, CdamLayout** d_layout) {
	*d_layout = CdamTMalloc(CdamLayout, 1, DEVICE_MEM);
	CdamMemcpy(*d_layout, layout, sizeof(CdamLayout), DEVICE_MEM, HOST_MEM);
	CdamLayoutNodalL2G(*d_layout) = CdamTMalloc(index_type, CdamMeshNumNode(mesh3d), DEVICE_MEM);

	index_type n = CdamLayoutNumExclusive(layout) + CdamLayoutNumShared(layout) + CdamLayoutNumGhosted(layout);
	CdamMemcpy(CdamLayoutNodalL2G(*d_layout), CdamLayoutNodalL2G(layout),
						 n * sizeof(index_type), DEVICE_MEM, DEVICE_MEM);
}

void VecDot(void* x, void* y, void* layout, void* result, void* results) {
	value_type* vx = (value_type*)x;
	value_type* vy = (value_type*)y;
	CdamLayout* lo = (CdamLayout*)layout;
	value_type* r = (value_type*)result;
	value_type* rs = (value_type*)results;

	value_type local_result[CDAM_MAX_NUM_COMPONENTS];

	index_type i, n = CdamLayoutNumComponents(lo);
	index_type num_owned = CdamLayoutNumOwned(lo);
	index_type num_ghost = CdamLayoutNumGhost(lo);
	index_type num = num_owned + num_ghost;
	index_type begin, end;
	for(i = 0; i < n; i++) {
		local_result[i] = 0.0;
		begin = CdamLayoutComponentOffset(lo)[i];
		end = CdamLayoutComponentOffset(lo)[i + 1];
		ddot(CdamLayoutNumOwned(lo) * (end - begin),
							vx + begin * num, 1,
							vy + begin * num, 1, local_result);																
	}
	MPI_Allreduce(MPI_IN_PLACE, local_result, n, MPI_VALUE_TYPE, MPI_SUM, MPI_COMM_WORLD);

	if(r) {
		*r = 0.0;
		for(i = 0; i < n; i++) {
			*r += local_result[i];
		}
	}
	if(rs) {
		for(i = 0; i < n; i++) {
			rs[i] = local_result[i];
		}
	}
}

void VecNorm(void* x, void* layout, void* result, void* results) {
	VecDot(x, x, layout, result, results);
}

__END_DECLS__
