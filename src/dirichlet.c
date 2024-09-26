#include <string.h>
#include "alloc.h"
#include "Mesh.h"
#include "parallel_matrix.h"
#include "dirichlet.h"

__BEGIN_DECLS__

Dirichlet* DirichletCreate(const CdamMesh* mesh, index_type face_ind, index_type shape) {
	Dirichlet* bc = CdamTMalloc(Dirichlet, 1, HOST_MEM);
	memset(bc, 0, sizeof(Dirichlet) + sizeof(BCType) * shape);
	bc->mesh = mesh;
	bc->face_ind = face_ind;
	bc->buffer_size = CdamMeshBoundNumNode(mesh, face_ind);
	bc->buffer = CdamTMalloc(index_type, bc->buffer_size, DEVICE_MEM);
	CdamMemcpy(bc->buffer, CdamMeshBoundNode(mesh, face_ind), bc->buffer_size * sizeof(index_type), DEVICE_MEM, HOST_MEM);
	bc->shape = shape;

	return bc;
}

void DirichletDestroy(Dirichlet* bc) {
	index_type shape = bc->shape;
	CdamFree(bc->buffer, bc->buffer_size * sizeof(index_type), DEVICE_MEM);
	bc->buffer_size = 0;
	CdamFree(bc, sizeof(Dirichlet) + shape * sizeof(BCType), HOST_MEM);
}

void ApplyBCVecNodalGPU(value_type*, index_type, const index_type*, index_type, index_type);

void DirichletApplyVec(Dirichlet* bc, value_type* b) {
	const CdamMesh* mesh = bc->mesh;
	index_type face_ind = bc->face_ind;
	index_type bound_num_node = CdamMeshBoundNumNode(mesh, face_ind);
	const index_type* bound_bnode = CdamMeshBoundNode(mesh, face_ind);

	for(index_type ic = 0; ic < bc->shape; ++ic) {
		if(bc->bctype[ic] == BC_STRONG) {
			ApplyBCVecNodalGPU(b, bound_num_node, bound_bnode, bc->shape, ic);
		}
	}
}

void GetRowFromNodeGPU(index_type, index_type*, index_type, index_type);
void GetNodeFromRowGPU(index_type, index_type*, index_type);

void DirichletApplyMat(Dirichlet* bc, void* A) {
	const CdamMesh* mesh = bc->mesh;
	index_type face_ind = bc->face_ind;
	index_type bound_num_node = CdamMeshBoundNumNode(mesh, face_ind);

	index_type* buffer = (index_type*)bc->buffer;

	for(index_type ic = 0; ic < bc->shape; ++ic) {
		if(bc->bctype[ic] == BC_STRONG) {
			GetRowFromNodeGPU(bound_num_node, buffer, (index_type)bc->shape, ic);
			// MatrixZeroRow(A, bound_num_node, buffer, 0, 1.0);
			CdamParMatZeroRow(A, bound_num_node, buffer, 1.0);
			GetNodeFromRowGPU(bound_num_node, buffer, bc->shape);
		}
	}
}

__END_DECLS__

