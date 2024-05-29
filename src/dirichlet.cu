#include "Mesh.h"
#include "matrix.h"
#include "dirichlet.h"

template <typename I, typename T> 
__global__ void apply_dirichlet_bc_vec(T* b, I n_bc, const I* bc_ind, const T* bc_val) {
	I i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_bc) return;
	if(bc_val) {
		b[bc_ind[i]] = bc_val[i];
	}
	else {
		b[bc_ind[i]] = T{0};
	}
}

template <typename I, typename T>
__global__ void foo(I n, const I* x, I incx,
										I a, I init,
										I* y, I incy) {
	I i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n) return;
	y[i * incy] = x[i * incx] * a + init;
}

template <typename I, typename T>
__global__ void GetRowFromNodeKernel(I n, I* row, I shape, I init) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n) return;
	row[i] = row[i] * shape + init;
}

template <typename I, typename T>
__global__ void GetNodeFromRowKernel(I n, I* node, I shape) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n) return;
	node[i] = node[i] / shape;
}

__BEGIN_DECLS__

Dirichlet* DirichletCreate(const Mesh3D* mesh, index_type face_ind, index_type shape) {
	Dirichlet* bc = (Dirichlet*)CdamMallocHost(sizeof(Dirichlet));
	memset(bc, 0, sizeof(Dirichlet));
	bc->mesh = mesh;
	const index_type* bound_ien = Mesh3DBoundIEN(mesh, face_ind);
	index_type bound_ien_size = Mesh3DBoundSize(mesh, face_ind);
	index_type num_unique = GetNumUniqueInd(bound_ien, bound_ien_size);
	index_type* h_bnode = (index_type*)CdamMallocHost(num_unique * sizeof(index_type));
	GetUniqueInd(bound_ien, bound_ien_size, h_bnode);

	bc->bnode = (index_type*)CdamMallocDevice(num_unique * sizeof(index_type));
	cudaMemcpy(bc->bnode, h_bnode, num_unique * sizeof(index_type), cudaMemcpyHostToDevice);

	bc->face_ind = face_ind;
	bc->shape = shape;
	bc->bctype = (BCtype*)CdamMallocHost(shape * sizeof(BCtype));

	CdamFreeHost(h_bnode, num_unique * sizeof(index_type));
	return bc;
}

void DirichletDestroy(Dirichlet* bc) {
	index_type shape = bc->shape;
	CdamFreeHost(bc, sizeof(Dirichlet) + shape * sizeof(BCtype));
}

void DirichletApplyVec(const Dirichlet* bc, value_type* b) {
	const Mesh3D* mesh = bc->mesh;
	index_type face_ind = bc->face_ind;
	index_type bound_bnode_size = Mesh3DBoundSize(mesh, face_ind);
	const index_type* bound_bnode = Mesh3DBoundNode(mesh, face_ind);

	int block_dim = 256;
	int grid_dim = (bound_bnode_size + block_dim - 1) / block_dim;
	for(index_type ic = 0; ic < bc->shape; ++ic) {
		if(bc->bctype[ic] == BC_STRONG) {
			GetRowFromNodeKernel<<<grid_dim, block_dim>>>(bound_bnode_size, (index_type*)bound_bnode, mesh->shape[face_ind], mesh->init[face_ind]);
		}
	}
}

__END_DECLS__

