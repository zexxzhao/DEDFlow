#include <string.h>
#include "common.h"

__BEGIN_DECLS__

#ifdef CDAM_USE_CUDA
__global__ void AddValuePrivate(void* dst, void* src, index_type block_length, index_type* displ, index_type count) {
	index_type i = blockIdx.x * blockDim.x + threadIdx.x;

	index_type dst_index, src_index; 
	value_type* dst_ptr = (value_type*)dst;
	value_type* src_ptr = (value_type*)src;
	block_length /= sizeof(value_type);


	if (i < count) {
		dst_index = displ[i % block_length] / sizeof(value_type) + i % block_length;
		src_index = i * block_length + i % block_length;
		atomicAdd(dst_ptr + dst_index, src_ptr[src_index]);
	}
}

__global__ void CopyValuePrivate(void* dst, void* src, index_type block_length, index_type* displ, index_type count) {
	index_type i = blockIdx.x * blockDim.x + threadIdx.x;

	index_type dst_index, src_index; 
	value_type* dst_ptr = (value_type*)dst;
	value_type* src_ptr = (value_type*)src;
	block_length /= sizeof(value_type);

	if (i < count) {
		dst_index = displ[i % block_length] / sizeof(value_type) + i % block_length;
		src_index = i * block_length + i % block_length;
		dst_ptr[dst_index] = src_ptr[src_index];
	}
}

#else
void AddValuePrivate(void* dst, void* src, index_type block_length, index_type* displ, index_type count) {
	index_type i, j;
	value_type* dst_ptr = (value_type*)dst;
	value_type* src_ptr = (value_type*)src;
	block_length /= sizeof(value_type);

	for (i = 0; i < count; i++) {
		for (j = 0; j < block_length; j++) {
			dst_ptr[displ[i] / sizeof(value_type) + j] += src_ptr[i * block_length + j];
		}
	}
}

void CopyValuePrivate(void* dst, void* src, index_type block_length, index_type* displ, index_type count) {
	index_type i, j;
	block_length /= sizeof(value_type);

	for (i = 0; i < count; i++) {
		memcpy((byte*)dst + displ[i], (byte*)src + i * block_length, block_length);
	}
}
#endif

__END_DECLS__
