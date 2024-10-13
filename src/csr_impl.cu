#ifdef CDAM_USE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#endif

#include "alloc.h"
// #include "Mesh.h"
#include "csr.h"

#define MAX_ROW_LENGTH 64
#ifdef CDAM_USE_CUDA
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
	if (i >= num_rows) return;
	Index start = row_ptr[i];
	Index len = row_ptr[i + 1] - start;
	for(Index j = 0; j < block_row; j++) {
		new_row_ptr[i * block_row + j] = start * block_row * block_col + j * block_col * len;
	}
}

template<typename Index>
__global__ void
SetColIndex(Index nnz, Index num_rows, 
						const Index* __restrict__ row_ptr, const Index* __restrict__ col_ind,
						Index block_row, Index block_col,
						const Index* __restrict__ new_row_ptr, Index* __restrict__ new_col_ind) {

	Index i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_rows) return;
	Index start = row_ptr[i], end = row_ptr[i + 1];
	Index len = end - start;

	for (Index j = 0; j < block_row; j++) {
		Index row = i * block_row + j;
		for(Index k = 0; k < len; k++) {
			Index col = col_ind[start + k];
			for(Index l = 0; l < block_col; l++) {
				new_col_ind[new_row_ptr[row] + k * block_col + l] = col * block_col + l;
			}
		}
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

template <typename I>
__global__ void
CSRAttrGetNZIndBatchedKernel(I num_row, I num_col, const I* row_ptr, const I* col_ind,
														 I batch_size, const I* row, const I* col, I* ind) {
	I i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batch_size) return;

	I row_start = row_ptr[row[i]];
	I row_end = row_ptr[row[i] + 1];

	for(I j = row_start; j < row_end; j++) {
		if(col[j] == col[i]) {
			ind[i] = j;
			return;
		}
	}
}

__BEGIN_DECLS__

typedef index_type index_t;
// void GenerateCSRFromMesh(const Mesh3D* mesh, CSRAttr* attr) {
// 	index_t num_node = Mesh3DNumNode(mesh);
// 	index_t num_elem = Mesh3DNumTet(mesh);
// 	const Mesh3DData* device = Mesh3DDevice(mesh);
// 	index_type* ien = device->ien;
// 	index_t* buffer = (index_t*)CdamMallocDevice(SIZE_OF(index_t) * num_node * MAX_ROW_LENGTH);
// 	/* Generate a vertex-to-vetex mapping */
// 	cudaMemset(buffer, 0, SIZE_OF(index_t) * num_node * MAX_ROW_LENGTH);
// 	fprintf(stderr, "Not implemented\n");
// 
// 	CdamFreeDevice(buffer, SIZE_OF(index_t) * num_node * MAX_ROW_LENGTH);
// }

void ExpandCSRByBlockSizeDevice(const CSRAttr* attr, CSRAttr* new_attr, index_t block_size[2]) {
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

	// CSRAttrRowPtr(new_attr) = (index_t*)CdamMallocDevice(SIZE_OF(index_t) * (CSRAttrNumRow(new_attr) + 1));
	// CSRAttrColInd(new_attr) = (index_t*)CdamMallocDevice(SIZE_OF(index_t) * CSRAttrNNZ(new_attr));
	// index_t* find_row = (index_t*)CdamMallocDevice(SIZE_OF(index_t) * nnz);

	int block_dim = 256;
	int block_num = (num_rows + block_dim - 1) / block_dim;
	// GetFindRowKernel<<<block_num, block_dim, 0, stream>>>(nnz, num_rows, row_ptr, find_row);
	SetRowLength<<<CEIL_DIV(num_rows, block_dim), block_dim, 0, stream>>>(num_rows, row_ptr, block_row, block_col, CSRAttrRowPtr(new_attr));
	SetColIndex<<<CEIL_DIV(num_rows, block_dim), block_dim, 0, stream>>>(nnz, num_rows, row_ptr, col_ind, block_row, block_col, CSRAttrRowPtr(new_attr), CSRAttrColInd(new_attr));

	// CdamFreeDevice(find_row, SIZE_OF(index_t) * nnz);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

}

void CSRAttrGetNZIndBatchedDevice(const CSRAttr* attr,
																 index_type batch_size, const index_type* row, const index_type* col,
																 index_type* ind) {
	index_type num_rows = CSRAttrNumRow(attr);
	index_type num_cols = CSRAttrNumCol(attr);
	index_type nnz = CSRAttrNNZ(attr);
	const index_type* row_ptr = CSRAttrRowPtr(attr);
	const index_type* col_ind = CSRAttrColInd(attr);

	int block_dim = 256;
	int grid_dim = (batch_size + block_dim - 1) / block_dim;
	CSRAttrGetNZIndBatchedKernel<index_type><<<grid_dim, block_dim>>>(num_rows, num_cols, row_ptr, col_ind, /* csr */
																																	  batch_size, row, col, ind);

}
__global__ void GetSubmatRowLength(index_type nrow, index_type ncol, index_type nnz,
																	 index_type* row_ptr, index_type* col_ind,
																	 index_type nr, index_type* row, index_type nc, index_type* col,
																	 index_type* new_row_ptr) {
	index_type i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > nr) return;

	index_type start = row_ptr[row[i]];
	index_type end = row_ptr[row[i] + 1];

	index_type len = 0;

	index_type j, k, ic;
	for(j = start; j < end; ++j) {
		ic = col_ind[j];
		for(k = 0; k < nc; ++k) {
			if(col[k] == ic) {
				++len;
				break;
			}
		}
	}
	new_row_ptr[i] = len;
}
__global__ void GetSubmatColInd(index_type nrow, index_type ncol, index_type nnz,
																index_type* row_ptr, index_type* col_ind,
																index_type nr, index_type* row, index_type nc, index_type* col,
																index_type* new_row_ptr, index_type* new_col_ind) {
	index_type i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > nr) return;

	index_type start = row_ptr[row[i]];
	index_type end = row_ptr[row[i] + 1];

	index_type len = 0;

	index_type j, k, ic;
	new_col_ind += new_row_ptr[i];
	for(j = start; j < end; ++j) {
		ic = col_ind[j];
		for(k = 0; k < nc; ++k) {
			if(col[k] == ic) {
				new_col_ind[len] = k;
				++len;
				break;
			}
		}
	}
}

void GenerateSubmatCSRAttrDevice(CSRAttr* attr, index_type nr, index_type* row,
																 index_type nc, index_type* col, CSRAttr** new_attr) {
	index_type num_rows = CSRAttrNumRow(attr);
	index_type num_cols = CSRAttrNumCol(attr);
	*new_attr = CdamTMalloc(CSRAttr, 1, HOST_MEM);
	CdamMemset(*new_attr, 0, sizeof(CSRAttr), HOST_MEM);
	CSRAttrNumRow(*new_attr) = nr;
	CSRAttrNumCol(*new_attr) = nc;
	CSRAttrRowPtr(*new_attr) = CdamTMalloc(index_type, nr + 1, DEVICE_MEM);
	CdamMemset(CSRAttrRowPtr(*new_attr), 0, sizeof(index_type) * (nr + 1), DEVICE_MEM);

	GetSubmatRowLength<<<CEIL_DIV(nr, 256), 256>>>(num_rows, num_cols, CSRAttrNNZ(attr),
																								 CSRAttrRowPtr(attr), CSRAttrColInd(attr),
																								 nr, row, nc, col, CSRAttrRowPtr(*new_attr) + 1);

	thrust::inclusive_scan(thrust::device_ptr<index_type>(CSRAttrRowPtr(*new_attr)),
												 thrust::device_ptr<index_type>(CSRAttrRowPtr(*new_attr) + nr + 1),
												 thrust::device_ptr<index_type>(CSRAttrRowPtr(*new_attr)));


	CdamMemcpy(&CSRAttrNNZ(*new_attr), CSRAttrRowPtr(*new_attr) + nr, sizeof(index_type), HOST_MEM, DEVICE_MEM);
	CSRAttrColInd(*new_attr) = CdamTMalloc(index_type, CSRAttrNNZ(*new_attr), DEVICE_MEM);
	GetSubmatColInd<<<CEIL_DIV(nr, 256), 256>>>(num_rows, num_cols, CSRAttrNNZ(attr),
																						  CSRAttrRowPtr(attr), CSRAttrColInd(attr),
																						  nr, row, nc, col, CSRAttrRowPtr(*new_attr), CSRAttrColInd(*new_attr));
}
															

__END_DECLS__
#endif
