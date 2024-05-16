
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <curand.h>
#include "alloc.h"
#include "color_impl.h"

#define COLOR_RANDOM_LB (0)
#define COLOR_RANDOM_UB (INT_MAX/2)

#define COLOR_MARKED (COLOR_RANDOM_UB + 1)

template <typename IndexType, int NSHL>
__global__ void GenerateV2EMapRowCountKernel(const IndexType* ien, IndexType num_elem, IndexType num_node, IndexType* row_ptr) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elem) return;
	#pragma unroll
	for(int j = 0; j < NSHL; ++j) {
		atomicAdd(row_ptr + ien[i * NSHL + j] + 1, 1);
	}
}

template <typename IndexType, int NSHL>
void GenerateV2EMapRowGPU(const IndexType* ien, IndexType num_elem, IndexType num_node, IndexType* row_ptr) {
	cudaMemset(row_ptr, 0, sizeof(IndexType) * (num_node + 1));
	int block_dim = 256;
	int grid_dim = (num_elem + block_dim - 1) / block_dim;

	GenerateV2EMapRowCountKernel<IndexType, NSHL><<<grid_dim, block_dim>>>(ien, num_elem, num_node, row_ptr);

	thrust::inclusive_scan(thrust::device, row_ptr, row_ptr + num_node + 1, row_ptr);
}

template <typename IndexType, int NSHL>
__global__ void GenerateV2EMapColKernel(const IndexType* ien, IndexType num_elem, IndexType num_node, const IndexType* row_ptr, IndexType* col_idx, IndexType* row_count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elem) return;
	#pragma unroll
	for(int j = 0; j < NSHL; ++j) {
		IndexType node = ien[i * NSHL + j];
		int offset = atomicAdd(row_count + node, (IndexType)1);
		col_idx[row_ptr[node] + offset] = i;
	}
}

template <typename IndexType, int NSHL>
void GenerateV2EMapColGPU(const IndexType* ien, IndexType num_elem, IndexType num_node, const IndexType* row_ptr, IndexType* col_idx) {
	int block_dim = 256;
	int grid_dim = (num_elem + block_dim - 1) / block_dim;

	IndexType* row_counter = (IndexType*)CdamMallocDevice(sizeof(IndexType) * num_node);
	cudaMemset(row_counter, 0, sizeof(IndexType) * num_node);

	GenerateV2EMapColKernel<IndexType, NSHL><<<grid_dim, block_dim>>>(ien, num_elem, num_node, row_ptr, col_idx, row_counter);

	CdamFreeDevice(row_counter, sizeof(IndexType) * num_node);
}

/* Jones-Plassman-Luby Algorithm */
template <typename IndexType, typename ColorType, int NSHL> __global__ void
ColorElementJPLKernel(const IndexType* ien, /* Element to Vertex */
											const IndexType* row_ptr, const IndexType* col_ind, /* Vertex to Element */
											ColorType max_color, ColorType* color, IndexType num_elem) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= num_elem) return;

	for (; i < num_elem; i += gridDim.x * blockDim.x) {
		b32 found_max = TRUE;

		ColorType ec = color[i];
		if (ec < 0) continue; /* Already colored */

		/* Loop all the connected elements */
		#pragma unroll
		for (int j = 0; j < NSHL; ++j) {
			IndexType node = ien[i * NSHL + j];
			IndexType start = row_ptr[node];
			IndexType end = row_ptr[node + 1];

			/* Loop all the connected elements */
			for (IndexType k = start; k < end; ++k) {
				IndexType elem = col_ind[k];
				if (ec < color[elem] && elem != i) {
					found_max = FALSE;
				}
			}
		}
		if(found_max) color[i] = COLOR_MARKED;
	}

}

template<typename ColorType>
struct ColorElementJPLReverseColorFunctor {
	ColorType max_color;
	ColorType c;
	ColorElementJPLReverseColorFunctor(ColorType c) : c(c) {}

	__host__ __device__ ColorType operator()(ColorType ec) const {
		i32 marked = (i32)(ec == COLOR_MARKED);
		return marked * c + (1 - marked) * ec;
	}

};

template <typename ColorType>
struct ColorElementJPLUncoloredFunctor {
	ColorElementJPLUncoloredFunctor() { }

	__host__ __device__ bool operator()(ColorType ec) const {
		return ec >= 0;
	}
};

template <typename IndexType, typename ColorType, int NSHL>
IndexType ColorElementJPLGPU(const IndexType* ien, /* Element to Vertex */
						const IndexType* row_ptr, const IndexType* col_ind, /* Vertex to Element */
						ColorType max_color, ColorType* color, IndexType num_elem) {
	int block_dim = 256;
	int grid_dim = (num_elem + block_dim - 1) / block_dim;

	IndexType left = num_elem;
	IndexType c = 0;

	GenerateRandomColor(color, num_elem, max_color);

	for(; c < max_color && left; c++) {
		ColorType reverse_color = -1 - ColorType(c);
		ColorElementJPLKernel<IndexType, ColorType, NSHL><<<grid_dim, block_dim>>>(ien, row_ptr, col_ind, max_color, color, num_elem);
		printf("c=%d, left = %d\n", c, left);
		thrust::transform(thrust::device, color, color + num_elem, color,
											ColorElementJPLReverseColorFunctor<ColorType>(reverse_color));
		left = thrust::count_if(thrust::device, color, color + num_elem, thrust::placeholders::_1 >= 0);
		cudaDeviceSynchronize();
	}
	thrust::transform(thrust::device, color, color + num_elem, color, thrust::placeholders::_1 * (-1) - 1);
	return c;
}

template <typename ColorType> struct GenerateRandomColorFunctor {
	ColorType lb, ub;
	GenerateRandomColorFunctor(ColorType lb, ColorType ub) : lb(lb), ub(ub) {}
	__host__ __device__ ColorType operator()(ColorType c) const {
		unsigned int val = *(unsigned int*)&c;
		return val % (ub - lb) + lb;
	}
};

__BEGIN_DECLS__

void GenerateV2EMapRowTetGPU(const index_type* ien, index_type num_elem, index_type num_node, index_type* row_ptr) {
	GenerateV2EMapRowGPU<index_type, 4>(ien, num_elem, num_node, row_ptr);
}
void GenerateV2EMapColTetGPU(const index_type* ien, index_type num_elem, index_type num_node, const index_type* row_ptr, index_type* col_idx) {
	GenerateV2EMapColGPU<index_type, 4>(ien, num_elem, num_node, row_ptr, col_idx);
}

void GenerateV2EMapRowPrismGPU(const index_type* ien, index_type num_elem, index_type num_node, index_type* row_ptr) {
	GenerateV2EMapRowGPU<index_type, 6>(ien, num_elem, num_node, row_ptr);
}
void GenerateV2EMapColPrismGPU(const index_type* ien, index_type num_elem, index_type num_node, const index_type* row_ptr, index_type* col_idx) {
	GenerateV2EMapColGPU<index_type, 6>(ien, num_elem, num_node, row_ptr, col_idx);
}

void GenerateV2EMapRowHexGPU(const index_type* ien, index_type num_elem, index_type num_node, index_type* row_ptr) {
	GenerateV2EMapRowGPU<index_type, 8>(ien, num_elem, num_node, row_ptr);
}

void GenerateV2EMapColHexGPU(const index_type* ien, index_type num_elem, index_type num_node, const index_type* row_ptr, index_type* col_idx) {
	GenerateV2EMapColGPU<index_type, 8>(ien, num_elem, num_node, row_ptr, col_idx);
}


void ColorElementJPLTetGPU(const index_type* ien, /* Element to Vertex */
						   const index_type* row_ptr, const index_type* col_ind, /* Vertex to Element */
						   color_t max_color, color_t* color, index_type num_elem) {
	ColorElementJPLGPU<index_type, color_t, 4>(ien, row_ptr, col_ind, max_color, color, num_elem);
}

void GenerateRandomColor(color_t* color, index_type n, color_t max_color) {
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	curandGenerate(gen, (unsigned int*)color, n);
	curandDestroyGenerator(gen);

	if(max_color == 0) {
		return;
	}

	thrust::transform(thrust::device, color, color + n, color, GenerateRandomColorFunctor<color_t>(COLOR_RANDOM_LB, COLOR_RANDOM_UB));
}


void GetMaxColorGPU(const color_t* color, index_type n, color_t* max_color) {
	thrust::device_ptr<const color_t> ptr(color);
	thrust::device_ptr<const color_t> max_ptr = thrust::max_element(ptr, ptr + n);
	*max_color = *max_ptr;
}
__END_DECLS__
