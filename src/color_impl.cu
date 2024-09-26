
#ifdef CDAM_USE_CUDA
#include <thrust/scan.h>
#include <cub/cub.cuh>
#include <curand.h>
#endif


#include "alloc.h"
#include "color_impl.h"
// #include <algorithm>
// #include <thrust/logical.h>
// #include <thrust/device_ptr.h>
// #include <thrust/execution_policy.h>
// #include <thrust/count.h>

__BEGIN_DECLS__
#ifdef CDAM_USE_CUDA
static __global__ void GenerateV2EMapRowCountKernel(const index_type* ien, index_type num_elem, index_type num_node, index_type* row_ptr, int NSHL) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elem) return;
	for(int j = 0; j < NSHL; ++j) {
		atomicAdd(row_ptr + ien[i * NSHL + j] + 1, 1);
	}
}

void GenerateV2EMapRowDevice(const index_type* ien, index_type num_elem, index_type num_node, index_type* row_ptr, int NSHL) {
	cudaMemset(row_ptr, 0, sizeof(index_type) * (num_node + 1));
	int block_dim = 256;
	int grid_dim = (num_elem + block_dim - 1) / block_dim;

	GenerateV2EMapRowCountKernel<index_type, NSHL><<<grid_dim, block_dim>>>(ien, num_elem, num_node, row_ptr);

	thrust::inclusive_scan(row_ptr, row_ptr + num_node + 1, row_ptr);
}

static __global__ void GenerateV2EMapColKernel(const index_type* ien, index_type num_elem, index_type num_node,
																							 const index_type* row_ptr, index_type* col_idx, index_type* row_count, int NSHL) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elem) return;
	for(int j = 0; j < NSHL; ++j) {
		index_type node = ien[i * NSHL + j];
		int offset = atomicAdd(row_count + node, (index_type)1);
		col_idx[row_ptr[node] + offset] = i;
	}
}

void GenerateV2EMapColDevice(const index_type* ien, index_type num_elem, index_type num_node, const index_type* row_ptr, index_type* col_idx, int NSHL) {
	int block_dim = 256;
	int grid_dim = (num_elem + block_dim - 1) / block_dim;

	index_type* row_counter = CdamTMalloc(index_type,  num_node, DEVICE_MEM);
	cudaMemset(row_counter, 0, sizeof(index_type) * num_node);

	GenerateV2EMapColKernel<index_type, NSHL><<<grid_dim, block_dim>>>(ien, num_elem, num_node, row_ptr, col_idx, row_counter);

	CdamFree(row_counter, sizeof(index_type) * num_node, DEVICE_MEM);
}

/* Jones-Plassman-Luby Algorithm */
static __global__ void
ColorElementJPLKernel(const index_type* ien, /* Element to Vertex */
											const index_type* row_ptr, const index_type* col_ind, /* Vertex to Element */
											index_type max_color, index_type* color, index_type num_elem, int NSHL) {
	const int NUM_THREAD = 192;
	const int MAX_NUM_ELEM_BLOCK = NUM_THREAD / 4;
	int num_elem_block = NUM_THREA / NSHL;
	__shared__ byte tmp[MAX_NUM_ELEM_BLOCK]; 

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadIdx.x < MAX_NUM_ELEM_BLOCK) 
		tmp[threadIdx.x] = 0;

	__syncthreads();
	if(i >= num_elem * NSHL) return;
	int iel = i / NSHL;
	int ishl = i % NSHL;

	index_type ec = color[iel];
	if (ec < 0) return; /* Already colored */

	/* Loop all the connected elements */
	index_type node = ien[iel * NSHL + ishl];
	index_type start = row_ptr[node];
	index_type end = row_ptr[node + 1];
	index_type k, elem;

	for (k = start; k < end; ++k) {
		elem = col_ind[k];
		if (ec < color[elem] || (ec == color[elem] && elem > iel)) {
			tmp[threadIdx.x / NSHL] = 1;
		}
	}

	__syncthreads();

	int oid = threadIdx.x + blockIdx.x * num_elem_block;
	if(threadIdx.x < num_elem_block && oid < num_elem) {
		if(!tmp[threadIdx.x]) {
			color[oid] = COLOR_MARKED;
		}
	}

	// for (int j = 0; j < NSHL; ++j) {
	// 	index_type node = ien[i * NSHL + j];
	// 	index_type start = row_ptr[node];
	// 	index_type end = row_ptr[node + 1];

	// 	/* Loop all the connected elements */
	// 	for (index_type k = start; k < end; ++k) {
	// 		index_type elem = col_ind[k];
	// 		if (ec < color[elem] && elem != i) {
	// 			found_max = FALSE;
	// 		}
	// 	}
	// }
	// if(found_max) color[i] = COLOR_MARKED;

}

struct ColorElementJPLReverseColorFunctor {
	index_type max_color;
	index_type c;
	ColorElementJPLReverseColorFunctor(index_type c) : c(c) {}

	__host__ __device__ index_type operator()(index_type ec) const {
		i32 marked = (i32)(ec == COLOR_MARKED);
		return marked * c + (1 - marked) * ec;
	}

};

struct ColorElementJPLUncoloredFunctor {
	ColorElementJPLUncoloredFunctor() { }

	__host__ __device__ bool operator()(index_type ec) const {
		return ec >= 0;
	}
};


__global__ void ReverseColorKernel(index_type* color, index_type c, index_type num_elem) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= num_elem) return;
	if(color[i] == COLOR_MARKED) {
		color[i] = c;
	}
}

__global__ void SetUpFlagKernel(index_type* color, u8* flag, index_type num_elem) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= num_elem) return;
	flag[i] = (u8)(color[i] >= 0);
}

__global__ void RecoverColorKernel(index_type* color, index_type num_elem) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= num_elem) return;
	color[i] = color[i] * (-1) - 1;
}


struct GenerateRandomColorFunctor {
	index_type lb, ub;
	GenerateRandomColorFunctor(index_type lb, index_type ub) : lb(lb), ub(ub) {}
	__host__ __device__ index_type operator()(index_type c) const {
		unsigned int val = *(unsigned int*)&c;
		return val % (ub - lb) + lb;
	}
};

index_type ColorElementJPL(const index_type* ien, /* Element to Vertex */
						index_type* row_ptr, index_type* col_ind, /* Vertex to Element */
						index_type max_color, index_type* color, index_type num_elem, int NSHL) {
	index_type left = num_elem;
	index_type c = 0;

	GenerateRandomColor(color, num_elem, max_color);

	u8 h_flag;

	int block_dim = 256*4;
	int grid_dim = (num_elem + block_dim - 1) / block_dim;


	u8* d_flag = CdamTMalloc(u8, num_elem + 1, DEVICE_MEM);
	u8* cub_reduce_buffer = NULL;
	size_t cub_reduce_buffer_size = 0;
	cub::DeviceReduce::Max(cub_reduce_buffer, cub_reduce_buffer_size, d_flag, d_flag + num_elem, num_elem);
	cub_reduce_buffer = CdamTMalloc(byte, cub_reduce_buffer_size, DEVICE_MEM);

	for(c = 0; c < max_color && left; c++) {
		index_type reverse_color = -1 - index_type(c);
		ColorElementJPLKernel<<<192, CEIL_DIV(num_elem * NSHL, 192)>>>(ien, row_ptr, col_ind, max_color, color, num_elem, NSHL);
		// printf("c=%d, left = %d\n", c, left);
		ReverseColorKernel<<<grid_dim, block_dim>>>(color, reverse_color, num_elem);
		SetUpFlagKernel<<<grid_dim, block_dim>>>(color, d_flag, num_elem);
		cub::DeviceReduce::Max(cub_reduce_buffer, cub_reduce_buffer_size, d_flag, d_flag + num_elem, num_elem);
		// cudaDeviceSynchronize();
		cudaMemcpyAsync(&h_flag, d_flag + num_elem, sizeof(u8), cudaMemcpyDeviceToHost);
		left = !!h_flag;
	}
	RecoverColorKernel<<<grid_dim, block_dim>>>(color, num_elem);
	CdamFreeDevice(d_flag, sizeof(u8) * (num_elem + 1), DEVICE_MEM);
	CdamFreeDevice(cub_reduce_buffer, cub_reduce_buffer_size, DEVICE_MEM);
	return c;
}

void GenerateRandomColorDevice(index_type* color, index_type n, index_type max_color) {
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	curandGenerate(gen, (unsigned int*)color, n);
	curandDestroyGenerator(gen);

	if(max_color == 0) {
		return;
	}

	thrust::transform(color, color + n, color, GenerateRandomColorFunctor<index_type>(COLOR_RANDOM_LB, COLOR_RANDOM_UB));
	for(index_type i = 0; i < n; i++) {
		color[i] = rand() % (COLOR_RANDOM_UB - COLOR_RANDOM_LB) + COLOR_RANDOM_LB;
	}
}

void GetMaxColorDevice(const index_type* color, index_type n, index_type* max_color, Arena scratch) {
	index_type* output = (index_type*)ArenaPush(sizeof(index_type), 1, scratch, 0);
	index_type* temp_buff = NULL;
	size_t temp_buff_size = 0;
	cub::DeviceReduce::Max(temp_buff, temp_buff_size, color, output, n);
	temp_buff = (index_type*)ArenaPush(sizeof(byte), temp_buff_size, scratch, 0);

	cub::DeviceReduce::Max(temp_buff, temp_buff_size, color, output, n);
	CdamMemcpy(max_color, output, sizeof(index_type), HOST_MEM, DEVICE_MEM);
}
#endif
__END_DECLS__

