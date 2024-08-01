#include "dirichlet.h"

template <typename I, typename T> 
__global__ void ApplyBCVecKernel(T* b, I n_bc, const I* bc_ind, const T* bc_val) {
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
__global__ void ApplyBCVecNodalKernel(T* b, I n_bc_node, const I* bc_node, I shape, I init) {
	I i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n_bc_node) return;
	I idx = bc_node[i] * shape + init;
	b[idx] = T{0};
}


template <typename I>
__global__ void GetRowFromNodeKernel(I n, I* row, I shape, I init) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n) return;
	row[i] = row[i] * shape + init;
}

template <typename I>
__global__ void GetNodeFromRowKernel(I n, I* node, I shape) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n) return;
	node[i] = node[i] / shape;
}

__BEGIN_DECLS__

void ApplyBCVecGPU(value_type* b, index_type n_bc, const index_type* bc_ind,
									 const value_type* bc_val) {
	int block_size = 256;
	int n_blocks = (n_bc + block_size - 1) / block_size;
	ApplyBCVecKernel<<<n_blocks, block_size>>>(b, n_bc, bc_ind, bc_val);
}

void ApplyBCVecNodalGPU(value_type* b, index_type n_bc_node, const index_type* bc_node,
												index_type shape, index_type init) {
	int block_size = 256;
	int n_blocks = (n_bc_node + block_size - 1) / block_size;
	ApplyBCVecNodalKernel<<<n_blocks, block_size>>>(b, n_bc_node, bc_node, shape, init);
}

void GetNodeFromRowGPU(index_type n, index_type* node, index_type shape) {
	int block_size = 256;
	int n_blocks = (n + block_size - 1) / block_size;
	GetNodeFromRowKernel<<<n_blocks, block_size>>>(n, node, shape);
}

void GetRowFromNodeGPU(index_type n, index_type* row, index_type shape, index_type init) {
	int block_size = 256;
	int n_blocks = (n + block_size - 1) / block_size;
	GetRowFromNodeKernel<<<n_blocks, block_size>>>(n, row, shape, init);
}
__END_DECLS__
