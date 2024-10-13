#include <string.h>
#ifdef CDAM_USE_CUDA
#include <cuda_runtime.h>
#include <cub/cub.cuh>
// #include <cub/device/device_transform.cuh>
#endif

#include "indexing.h"

#ifdef CDAM_USE_CUDA

struct IsEqualTo {
	index_type value;
	__host__ __device__ IsEqualTo(index_type v) : value(v) {}
	__host__ __device__ __forceinline__ bool operator()(index_type a) {
		return a == value;
	}
};

__global__ void FilterArrayByValueKernel(void* data, int elem_size, int size,
		void* value, int* out) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid * elem_size < size) {
		for(int i = 0; i < elem_size; ++i) {
			if (((char*)data)[tid * elem_size + i] != ((char*)value)[i]) {
				out[tid] = 0;
				return;
			}
		
		}
		out[tid] = 1;
	}
}
template<typename T>
__global__ void TransformByRange(index_type n, index_type* data, T* mask, index_type start, index_type end) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n) {
		mask[tid] = static_cast<T>((data[tid] >= start && data[tid] < end) ? 1 : 0);
	}
}
#endif





__BEGIN_DECLS__
#ifdef CDAM_USE_CUDA
index_type CountValueI(index_type* data, index_type size, index_type value, Arena scratch) {
	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	index_type* d_mask = (index_type*)ArenaPush(sizeof(index_type), size + 1, &scratch, 0);
	TransformByRange<<<CEIL_DIV(size, 256), 256>>>(size, data, d_mask, value, value + 1);

	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_mask, d_mask + size, size);
	d_temp_storage = ArenaPush(1, temp_storage_bytes, &scratch, 0);
	index_type count;
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_mask, d_mask + size, size);

	CdamMemcpy(&count, d_mask + size, sizeof(index_type), HOST_MEM, DEVICE_MEM);
	return count;
}

index_type CountValueImp(void* data, index_type elem_size, index_type n, void* value, Arena scratch) {
	int* d_out = (int*)ArenaPush(sizeof(int), n / elem_size, &scratch, 0);
	int* d_value = (int*)ArenaPush(elem_size, 1, &scratch, 0);
	int num_thread = 256;
	int num_block = CEIL_DIV(n / elem_size, num_thread);

	CdamMemcpy(d_value, value, elem_size, DEVICE_MEM, HOST_MEM);
	FilterArrayByValueKernel<<<num_block, num_thread>>>((void*)data, elem_size, n, value, d_out);

	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	int* d_num_selected_out = (int*)ArenaPush(sizeof(int), 1, &scratch, 0);
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_out, d_num_selected_out, n / elem_size);
	d_temp_storage = ArenaPush(1, temp_storage_bytes, &scratch, 0);
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_out, d_num_selected_out, n / elem_size);
	int h_num_selected;
	CdamMemcpy(&h_num_selected, d_num_selected_out, sizeof(int), HOST_MEM, DEVICE_MEM);
	return (index_type)h_num_selected;
}

void SortByKeyI(index_type* key, index_type* value, index_type n, Arena scratch) {
	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	index_type* d_key_out = (index_type*)ArenaPush(sizeof(index_type), n, &scratch, 0);
	index_type* d_value_out = (index_type*)ArenaPush(sizeof(index_type), n, &scratch, 0);
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, key, d_key_out, value, d_value_out, n);
	d_temp_storage = ArenaPush(1, temp_storage_bytes, &scratch, 0);
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, key, d_key_out, value, d_value_out, n);
	CdamMemcpy(value, d_value_out, n * sizeof(index_type), DEVICE_MEM, DEVICE_MEM);
}


index_type CountIndexByRangeGPU(index_type n, index_type* index, index_type start, index_type end) {
	byte* masked_val = CdamTMalloc(byte, n + 1, DEVICE_MEM);
	// auto op = [start, end] __device__ (index_type val) {
	// 	return (val >= start && val < end) ? 1 : 0;
	// };
	// cub::DeviceTransform::Transform(index, masked_val, n, op);
	int num_thread = 256;
	int num_block = CEIL_DIV(n, num_thread);
	TransformByRange<<<num_block, num_thread>>>(n, index, masked_val, start, end);

	size_t temp_buff_size = 0;
	index_type ret = 0;
	cub::DeviceReduce::Sum(nullptr, temp_buff_size, masked_val, masked_val + n, n);
	byte* temp_buff = CdamTMalloc(byte, temp_buff_size, DEVICE_MEM);
	cub::DeviceReduce::Sum(temp_buff, temp_buff_size, masked_val, masked_val + n, n);
	CdamMemcpy(&ret, masked_val + n, sizeof(byte), HOST_MEM, DEVICE_MEM);
	CdamFree(masked_val, (n + 1) * sizeof(byte), DEVICE_MEM);
	CdamFree(temp_buff, temp_buff_size, DEVICE_MEM);
	return ret;
}
static __global__ void IotaKernel(index_type* data, index_type n, index_type start, index_type step) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n) {
		data[tid] = start + tid * step;
	}
}
void Iota(index_type* data, index_type n, index_type start, index_type step) {
	int num_thread = 256;
	int num_block = CEIL_DIV(n, num_thread);
	IotaKernel<<<num_block, num_thread>>>(data, n, start, step);
}

static __global__ void GenMaskByRangeKernel(index_type nrow, index_type nshl, index_type *input,
																						byte* mask, index_type start, index_type end) {

	const int NUM_THREAD = 192;
	__shared__ byte tmp[NUM_THREAD];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int n_per_block = NUM_THREAD / nshl;
	int oid = threadIdx.x + blockIdx.x * n_per_block;
	int i;
	byte output = 0;
	if(tid >= nrow * nshl) return;
	
	index_type v = input[tid];
	tmp[threadIdx.x] = (v >= start && v < end) ? 1 : 0;

	__syncthreads();

	if(threadIdx.x < n_per_block && oid < nrow) {
		for(int i = 0; i < nshl; ++i) {
			output |= tmp[threadIdx.x * nshl + i] * (1 << i);
		}
		mask[oid] = output;
	}
}

void GenMaskByRange(index_type nrow, index_type ncol, index_type* input,
										byte* mask, index_type start, index_type end, Arena scratch) {
	int num_thread = 192;
	int num_block = CEIL_DIV(nrow * ncol, num_thread);
	GenMaskByRangeKernel<<<num_block, num_thread>>>(nrow, ncol, input, mask, start, end);
}

static __global__ void RestrictVecKernel(void* dst, void* src, index_type n, index_type* index, size_t elem_size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	byte* d = (byte*)dst;
	byte* s = (byte*)src;

	if(tid >= n * elem_size) return;

	int i = tid / elem_size;
	int j = tid % elem_size;

	d += i * elem_size;
	s += index[i] * elem_size;
	d[j] = s[j];
}

void RestrictVec(void* dst, void* src, index_type n, index_type* index, size_t elem_size) {
	int num_thread = 256;
	int num_block = CEIL_DIV(n * elem_size, num_thread);
	RestrictVecKernel<<<num_block, num_thread>>>(dst, src, n, index, elem_size);
}

static __global__ void ProlongateVecKernel(void* dst, void* src, index_type n, index_type* index, size_t elem_size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	byte* d = (byte*)dst;
	byte* s = (byte*)src;

	if(tid >= n * elem_size) return;

	int i = tid / elem_size;
	int j = tid % elem_size;

	d += index[i] * elem_size;
	s += i * elem_size;
	d[j] = s[j];
}

void ProlongateVec(void* dst, void* src, index_type n, index_type* index, size_t elem_size) {
	int num_thread = 256;
	int num_block = CEIL_DIV(n * elem_size, num_thread);

	ProlongateVecKernel<<<num_block, num_thread>>>(dst, src, n, index, elem_size);
}

static __global__ void RestrictAddVecStridedKernel(value_type* dst, index_type dst_stride,
																									value_type* src, index_type src_stride,
																									index_type n, index_type* index, index_type block_size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= n * block_size) return;

	int i = tid / block_size;
	int j = tid % block_size;

	dst[i * dst_stride + j] += src[index[i] * src_stride + j];

}

void RestrictAddVecStrided(value_type* dst, index_type dst_stride,
													 value_type* src, index_type src_stride,
													 index_type n, index_type* index, index_type block_size) {
	int num_thread = 256;
	int num_block = CEIL_DIV(n * block_size, num_thread);
	RestrictAddVecStridedKernel<<<num_block, num_thread>>>(dst, dst_stride, src, src_stride, n, index, block_size);
}

static __global__ void PrologateAddVecStridedKernel(value_type* dst, index_type dst_stride,
																									value_type* src, index_type src_stride,
																									index_type n, index_type* index, index_type block_size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= n * block_size) return;

	int i = tid / block_size;
	int j = tid % block_size;

	dst[index[i] * dst_stride + j] += src[i * src_stride + j];
}

void PrologateAddVecStrided(value_type* dst, index_type dst_stride,
													 value_type* src, index_type src_stride,
													 index_type n, index_type* index, index_type block_size) {
	int num_thread = 256;
	int num_block = CEIL_DIV(n * block_size, num_thread);
	PrologateAddVecStridedKernel<<<num_block, num_thread>>>(dst, dst_stride, src, src_stride, n, index, block_size);
}

#else
index_type CountValueI(index_type* data, index_type size, index_type value, Arena scratch) {
	index_type count = 0;
	for (index_type i = 0; i < size; ++i) {
		if (data[i] == value) {
			++count;
		}
	}
	return count;
}
index_type CountValueImp(void* data, index_type elem_size, index_type n, void* value, Arena scratch) {
	index_type count = 0;
	for (index_type i = 0; i < n; i += elem_size) {
		if (memcmp((char*)data + i, value, elem_size) == 0) {
			++count;
		}
	}
	return count;
}

struct QsortData {
	void* beg;
	index_type* key;
	index_type elem_size;
};

struct QsortData* qsort_data;
static int CompareQsortData(void* a, void* b) {
	index_type ia = (char*)a - (char*)qsort_data->beg;
	ia /= qsort_data->elem_size;
	index_type ib = (char*)b - (char*)qsort_data->beg;
	ib /= qsort_data->elem_size;

	return qsort_data->key[ia] - qsort_data->key[ib];
}

void SortByKeyImp(void* d_value, index_type* d_key, index_type elem_size, index_type n, Arena scratch) {
	struct QsortData qd;
	qd.beg = d_value;
	qd.key = d_key;
	qd.elem_size = elem_size;
	qsort_data = &qd;

	qsort(d_value, n, elem_size, CompareQsortData);
}

index_type CountIndexByRangeGPU(index_type n, index_type* index, index_type start, index_type end) {
	index_type count = 0;
	for (index_type i = 0; i < n; ++i) {
		if (index[i] >= start && index[i] < end) {
			++count;
		}
	}
	return count;
}

void Iota(index_type* data, index_type n, index_type start, index_type step) {
	for (index_type i = 0; i < n; ++i) {
		data[i] = start + i * step;
	}
}
void RestrictVec(void* dst, void* src, index_type n, index_type* index, size_t elem_size) {
	byte* d = (byte*)dst;
	byte* s = (byte*)src;
	for (index_type i = 0; i < n; ++i) {
		CdamMemcpy(d + i * elem_size, s + index[i] * elem_size, elem_size, HOST_MEM, HOST_MEM);
	}
}
void ProlongateVec(void* dst, void* src, index_type n, index_type* index, size_t elem_size) {
	byte* d = (byte*)dst;
	byte* s = (byte*)src;
	for(index_type i = 0; i < n; ++i) {
		CdamMemcpy(d + index[i] * elem_size, s + i * elem_size, elem_size, HOST_MEM, HOST_MEM);
	}
}


void RestrictAddVecStrided(value_type* dst, index_type dst_stride,
													 value_type* src, index_type src_stride,
													 index_type n, index_type* index, index_type block_size) {
	
	int i;
	for(i = 0; i < n; ++i) {
		daxpy(block_size, 1.0, src + index[i] * src_stride, 1, dst + i * dst_stride, 1);
	}
}

void ProlongateAddVecStrided(value_type* dst, index_type dst_stride,
														 value_type* src, index_type src_stride,
														 index_type n, index_type* index, index_type block_size) {
	int i;
	for(i = 0; i < n; ++i) {
		daxpy(block_size, 1.0, src + i * src_stride, 1, dst + index[i] * dst_stride, 1);
	}
}

#endif
__END_DECLS__

