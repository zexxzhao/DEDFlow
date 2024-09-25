#include <string.h>

#include "alloc.h"
#include "blas.h"
#include "Mesh.h"
#include "json.h"
#include "vec.h"

__BEGIN_DECLS__


void CdamLayoutCreate(CdamLayout** layout, void* config) {
	*layout = CdamTMalloc(CdamLayout, 1, HOST_MEM);
	CdamMemset(*layout, 0, sizeof(CdamLayout), HOST_MEM);
	cJSON* json = (cJSON*)config;
	b32 has_ns = JSONGetItem(json, "VMS.IncompressibleNS.Included")->valuedouble > 0.0;
	b32 has_t = JSONGetItem(json, "VMS.Temperature.Included")->valuedouble > 0.0;
	b32 has_phi = JSONGetItem(json, "VMS.Levelset.Included")->valuedouble > 0.0;

	index_type num_components = 0;

	if(has_ns) {
		num_components++;
		CdamLayoutComponentOffset(*layout)[num_components] = CdamLayoutComponentOffset(*layout)[num_components - 1] + 3;
		num_components++;
		CdamLayoutComponentOffset(*layout)[num_components] = CdamLayoutComponentOffset(*layout)[num_components - 1] + 1;
	}

	if(has_t) {
		num_components++;
		CdamLayoutComponentOffset(*layout)[num_components] = CdamLayoutComponentOffset(*layout)[num_components - 1] + 1;
	}

	if(has_phi) {
		num_components++;
		CdamLayoutComponentOffset(*layout)[num_components] = CdamLayoutComponentOffset(*layout)[num_components - 1] + 1;
	}

	CdamLayoutNumComponents(*layout) = num_components;

}

void CdamLayoutDestroy(CdamLayout* layout) {
	CdamFree(layout, sizeof(CdamLayout), HOST_MEM);
}

void CdamLayoutSetup(CdamLayout* layout, void* mesh) {
	CdamMesh* mesh3d = (CdamMesh*)mesh;
	CdamLayoutNumOwned(layout) = CdamMeshLocalNodeEnd(mesh3d) - CdamMeshLocalNodeBegin(mesh3d);
	CdamLayoutNumGhost(layout) = CdamMeshNumNode(mesh3d) - CdamLayoutNumOwned(layout);
}

void VecDot(void* x, void* y, void* layout, void* result, void* results) {
	value_type* vx = (value_type*)x;
	value_type* vy = (value_type*)y;
	CdamLayout* lo = (CdamLayout*)layout;
	value_type* r = (value_type*)result;
	value_type* rs = (value_type*)results;

	value_type local_result[CDAM_VEC_MAX_NUM_COMPONENTS];

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
