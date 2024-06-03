#include <string.h>
#include "alloc.h"
#include "Mesh.h"
#include "matrix.h"
#include "dirichlet.h"

__BEGIN_DECLS__

Dirichlet* DirichletCreate(const Mesh3D* mesh, index_type face_ind, index_type shape) {
	Dirichlet* bc = (Dirichlet*)CdamMallocHost(SIZE_OF(Dirichlet) + SIZE_OF(BCType) * shape);
	memset(bc, 0, SIZE_OF(Dirichlet));
	bc->mesh = mesh;
	bc->face_ind = face_ind;
	bc->buffer_size = Mesh3DBoundNumNode(mesh, face_ind);
	bc->buffer = (index_type*)CdamMallocDevice(bc->buffer_size * SIZE_OF(index_type));
	cudaMemcpy(bc->buffer, Mesh3DBoundNode(mesh, face_ind), bc->buffer_size * SIZE_OF(index_type), cudaMemcpyHostToDevice);
	bc->shape = shape;

	return bc;
}

void DirichletDestroy(Dirichlet* bc) {
	index_type shape = bc->shape;
	CdamFreeDevice(bc->buffer, bc->buffer_size * SIZE_OF(index_type));
	bc->buffer_size = 0;
	CdamFreeHost(bc, SIZE_OF(Dirichlet) + shape * SIZE_OF(BCType));
}

void ApplyBCVecNodalGPU(value_type*, index_type, const index_type*, index_type, index_type);

void DirichletApplyVec(Dirichlet* bc, value_type* b) {
	const Mesh3D* mesh = bc->mesh;
	index_type face_ind = bc->face_ind;
	index_type bound_num_node = Mesh3DBoundNumNode(mesh, face_ind);
	const index_type* bound_bnode = Mesh3DBoundNode(mesh, face_ind);

	for(index_type ic = 0; ic < bc->shape; ++ic) {
		if(bc->bctype[ic] == BC_STRONG) {
			ApplyBCVecNodalGPU(b, bound_num_node, bound_bnode, bc->shape, ic);
		}
	}
}

void GetRowFromNodeGPU(index_type, index_type*, index_type, index_type);
void GetNodeFromRowGPU(index_type, index_type*, index_type);

void DirichletApplyMat(Dirichlet* bc, Matrix* A) {
	const Mesh3D* mesh = bc->mesh;
	index_type face_ind = bc->face_ind;
	index_type bound_num_node = Mesh3DBoundNumNode(mesh, face_ind);

	index_type* buffer = (index_type*)bc->buffer;

	for(index_type ic = 0; ic < bc->shape; ++ic) {
		if(bc->bctype[ic] == BC_STRONG) {
			GetRowFromNodeGPU(bound_num_node, buffer, (index_type)bc->shape, ic);
			MatrixZeroRow(A, bound_num_node, buffer, 0, 1.0);
			GetNodeFromRowGPU(bound_num_node, buffer, bc->shape);
		}
	}
}

__END_DECLS__

