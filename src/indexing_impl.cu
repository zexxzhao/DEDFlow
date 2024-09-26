#include <string.h>
#include "indexing.h"
#ifdef CDAM_USE_CUDA
#include <cuda_runtime.h>
#include <cub/cub.cuh>

struct IsEqualTo {
	index_type value;
	__host__ __device__ IsEqualTo(index_type v) : value(v) {}
	__host__ __device__ __forceinline__ bool operator()(index_type a) const {
		return a == value;
	}
};

__global__ void FilterArrayByValueKernel(const void* data, int elem_size, int size,
		const void* value, int* out) {
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
#endif





__BEGIN_DECLS__
#ifdef CDAM_USE_CUDA
index_type CountValueI(const index_type* data, index_type size, index_type value, Arena scratch) {
	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	int* d_num_selected_out = ArenaPush(sizeof(int), 1, &scratch, 0);

	IsEqualTo pred(value);

	cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, data, d_num_selected_out, size, pred);

	d_tmp_storage = ArenaPush(1, temp_storage_bytes, &scratch, 0);

	cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, data, d_num_selected_out, size, pred);

	int num_selected;
	CdamMemcpy(&num_selected, d_num_selected_out, sizeof(int), HOST_MEM, DEVICE_MEM);

	return (index_type)num_selected;
}

index_type CountValueImp(const void* data, index_type elem_size, index_type n, void* value, Arena scratch) {
	int* d_out = ArenaPush(sizeof(int), n / elem_size, &scratch, 0);
	int* d_value = ArenaPush(elem_size, 1, &scratch, 0);
	int num_thread = 256;
	int num_block = CEIL_DIV(n / elem_size, num_thread);

	CdamMemcpy(d_value, value, elem_size, DEVICE_MEM, HOST_MEM);
	FilterArrayByValueKernel<<<num_block, num_thread>>>((const void*)data, elem_size, n, value, d_out);

	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	int* d_num_selected_out = ArenaPush(sizeof(int), 1, &scratch, 0);
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
	index_type* d_key_out = ArenaPush(sizeof(index_type), n, &scratch, 0);
	index_type* d_value_out = ArenaPush(sizeof(index_type), n, &scratch, 0);
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, key, d_key_out, value, d_value_out, n);
	d_temp_storage = ArenaPush(1, temp_storage_bytes, &scratch, 0);
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, key, d_key_out, value, d_value_out, n);
	CdamMemcpy(value, d_value_out, n * sizeof(index_type), DEVICE_MEM, DEVICE_MEM);
}

index_type CountIndexByRangeGPU(index_type n, index_type* index, index_type start, index_type end) {
	index_type* masked_val = CdamTMalloc(index_type, n + 1, DEVICE_MEM);
	cub::DeviceTransform::Transform(index, masked_val, n, [start, end] __device__ (index_type val) {
		return (val >= start && val < end) ? 1 : 0;
	});

	size_t temp_buff_size = 0;
	index_type ret = 0;
	cub::DeviceReduce::Sum(nullptr, temp_buff_size, masked_val, masked_val + n, n);
	index_type* temp_buff = CdamTMalloc(index_type, temp_buff_size, DEVICE_MEM);
	cub::DeviceReduce::Sum(temp_buff, temp_buff_size, masked_val, masked_val + n, n);
	CdamMemcpy(&ret, masked_val + n, sizeof(index_type), HOST_MEM, DEVICE_MEM);
	CdamFree(masked_val, (n + 1) * sizeof(index_type), DEVICE_MEM);
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

static __global__ void GenMaskByRange(index_type nrow, index_type ncol, index_type *input,
																			byte* mask, index_type start, index_type end) {
	const int NUM_THREAD = 192;
	__shared__ byte tmp[NUM_THREAD];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int n_per_block = NUM_THREAD / nshl;
	int oid = threadIdx.x + blockIdx.x * n_per_block;
	int i;
	byte ouput = 0;
	if(tid >= nrow * ncol) return;
	
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
	GenMaskByRange<<<num_block, num_thread>>>(nrow, ncol, input, mask, start, end);
}
#else
index_type CountValueI(const index_type* data, index_type size, index_type value, Arena scratch) {
	index_type count = 0;
	for (index_type i = 0; i < size; ++i) {
		if (data[i] == value) {
			++count;
		}
	}
	return count;
}
index_type CountValueImp(const void* data, index_type elem_size, index_type n, void* value, Arena scratch) {
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
static int CompareQsortData(const void* a, const void* b) {
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
#endif
__END_DECLS__

