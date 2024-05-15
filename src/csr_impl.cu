#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "alloc.h"
#include "Mesh.h"
#include "csr.h"

#define MAX_ROW_LENGTH 64
template<typename Index>
__global__ void
GetFindRowKernel(Index nnz, Index num_rows,
								 const Index* __restrict__ row_ptr,
								 Index* __restrict__ find_row) {
	Index i = blockIdx.x * blockDim.x + threadIdx.x;
	while(i < num_rows) {
		for(Index j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
			find_row[j] = i;
		}
		i += blockDim.x * gridDim.x;
	}
}

template<typename Index>
__global__ void
SetRowLength(Index num_rows, const Index* __restrict__ row_ptr,
						 Index block_row, Index block_col,
						 Index* __restrict__ new_row_ptr) {
	Index i = blockIdx.x * blockDim.x + threadIdx.x;
	Index row = i / num_rows;
	Index lane = i % num_rows;
	
	if (i < num_rows) {
		Index start = row_ptr[i];
		Index len = row_ptr[i + 1] - start;
		new_row_ptr[i + 1] = start * block_row * block_col + lane * block_col * len;
	}
}

template<typename Index>
__global__ void
SetColIndex(Index nnz, Index num_rows, const Index* __restrict__ row_ptr, const Index* __restrict__ col_ind,
						Index block_row, Index block_col,
						const Index* __restrict__ find_row,
						const Index* __restrict__ new_row_ptr, Index* __restrict__ new_col_ind) {
	Index i = blockIdx.x * blockDim.x + threadIdx.x;
	while(i < nnz * block_row * block_col) {
		Index row = find_row[i / block_row];

		Index col = col_ind[i / (block_row* block_col)];
		
		new_col_ind[i] = col * block_col + i % block_col;

		i += blockDim.x * gridDim.x;
	}
}

/*
template<typename Index, int NSHL=4>
__global__ void
GenerateV2VMap(Index num_elem, const Index* __restrict__ ien, Index* __restrict__ v2e_map) {
	Index b_start = blockIdx.x * blockDim.x;
	Index b_end = min(b_start + blockDim.x, num_elem);
	Index i = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ Index local_ien[NSHL * blockDim.x];
	__shared__ Index local_v2v_map[MAX_BATCH_LENGTH * blockDim.x];
	while(i < num_elem) {
		for(Index j = 0; j < NSHL; ++j) {
			local_ien[threadIdx.x + j * (b_end - b_start)] = ien[i + j * (b_end - b_start)];
		}
		for(Index j = 0; j < MAX_BATCH_LENGTH; ++j) {
			local_v2v_map[threadIdx.x + j * blockDim.x] = 0;
		}
		__syncthreads();
		

		v2e_map[i * MAX_BATCH_LENGTH] = 0;
		for(Index j = 0; j < NSHL; j++) {
			Index node = elem[j];
			Index c = atomicAdd(v2e_map + node * MAX_BATCH_LENGTH, 1);
			v2e_map[node * MAX_BATCH_LENGTH + c + 1] = i;
		}
		i += blockDim.x * gridDim.x;
	}
}
*/


__BEGIN_DECLS__

typedef csr_index_type index_t;
void GenerateCSRFromMesh(const Mesh3D* mesh, CSRAttr* attr) {
	index_t num_node = Mesh3DNumNode(mesh);
	index_t num_elem = Mesh3DNumTet(mesh);
	const Mesh3DData* device = Mesh3DDevice(mesh);
	u32* ien = device->ien;
	index_t* buffer = (index_t*)CdamMallocDevice(sizeof(index_t) * num_node * MAX_ROW_LENGTH);
	/* Generate a vertex-to-vetex mapping */
	cudaMemset(buffer, 0, sizeof(index_t) * num_node * MAX_ROW_LENGTH);
	fprintf(stderr, "Not implemented\n");

	CdamFreeDevice(buffer, sizeof(index_t) * num_node * MAX_ROW_LENGTH);
}

void ExpandCSRByBlockSize(const CSRAttr* attr, CSRAttr* new_attr, index_t block_size[2]) {
	index_t num_rows = CSRAttrNumRow(attr);
	index_t num_cols = CSRAttrNumCol(attr);
	index_t nnz = CSRAttrNNZ(attr);
	index_t* row_ptr = CSRAttrRowPtr(attr);
	index_t* col_ind = CSRAttrColInd(attr);

	index_t block_row = block_size[0];
	index_t block_col = block_size[1];
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	CSRAttrNumRow(new_attr) = num_rows * block_row;
	CSRAttrNumCol(new_attr) = num_cols * block_col;
	CSRAttrNNZ(new_attr) = nnz * block_row * block_col;

	CSRAttrRowPtr(new_attr) = (index_t*)CdamMallocDevice(sizeof(index_t) * (CSRAttrNumRow(new_attr) + 1));
	CSRAttrColInd(new_attr) = (index_t*)CdamMallocDevice(sizeof(index_t) * CSRAttrNNZ(new_attr));
	index_t* find_row = (index_t*)CdamMallocDevice(sizeof(index_t) * nnz);

	int block_dim = 256;
	int block_num = (num_rows + block_dim - 1) / block_dim;
	GetFindRowKernel<<<block_num, block_dim, 0, stream>>>(nnz, num_rows, row_ptr, find_row);
	SetRowLength<<<block_num, block_dim, 0, stream>>>(num_rows, row_ptr, block_row, block_col, CSRAttrRowPtr(new_attr));
	SetColIndex<<<block_num, block_dim, 0, stream>>>(nnz, num_rows, row_ptr, col_ind, block_row, block_col, find_row, CSRAttrRowPtr(new_attr), CSRAttrColInd(new_attr));

	CdamFreeDevice(find_row, sizeof(index_t) * nnz);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

}


__END_DECLS__
