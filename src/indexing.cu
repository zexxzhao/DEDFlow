#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <cub/device/device_reduce.cuh>
#include "alloc.h"
#include "color.h"

template <typename IndexType, typename ValueType>
struct CountingFunctor {
	ValueType val;
	CountingFunctor(ValueType v) : val(v) {}

	__device__ IndexType operator()(IndexType a, ValueType b) const {
		return a + (b == val);
	}
};

template <typename IndexType, typename ValueType>
IndexType CountValueLegacy(const ValueType* data, IndexType n, ValueType value) {

		return thrust::count(data, data + n, value);
}


template <typename IndexType, typename ValueType>
IndexType CountValue(const ValueType* data, IndexType n, ValueType value, void* buffer) {

	size_t buffer_size_used = 0;
	IndexType init = 0;
	if(buffer == NULL) {
		IndexType* output = (IndexType*)CdamMallocDevice(sizeof(IndexType) * 1024);
		cub::DeviceReduce::Reduce(NULL, buffer_size_used, data, output, n, CountingFunctor<IndexType, ValueType>(value), init);
		CdamFreeDevice(output, sizeof(IndexType) * 1024);
		return (IndexType)buffer_size_used;
	}
	
	IndexType* output = (IndexType*)CdamMallocDevice(sizeof(IndexType) * 1024);
	cub::DeviceReduce::Reduce(buffer, buffer_size_used, data, output, n, CountingFunctor<IndexType, ValueType>(value), init);
	int sum;
	cudaMemcpy(&sum, output, sizeof(int), cudaMemcpyDeviceToHost);
	CdamFreeDevice(output, sizeof(IndexType) * 1024);
	return (IndexType)sum;
}

template <typename IndexType, typename ValueType>
struct EqualToValue {
	const ValueType* _data;
	ValueType _val;

	EqualToValue(const ValueType* data, ValueType val) : _data(data), _val(val) {}
	__host__ __device__
	bool operator()(IndexType i) const {
		return _data[i] == _val;
	}
};

/**
 * Find all indices of a given value in an array.
 * @param data The array to search.
 * @param n The size of the array.
 * @param value The value to search for.
 * @param result The array to store the indices of the value.
 */
template <typename IndexType, typename ValueType>
void FindValue(const ValueType* data, IndexType n, ValueType value, IndexType* result) {
	thrust::device_ptr<const ValueType> data_ptr(data);
	thrust::device_ptr<IndexType> result_ptr(result);

	EqualToValue<IndexType, ValueType> pred(data, value);

	thrust::copy_if(thrust::make_counting_iterator<IndexType>(0),
									thrust::make_counting_iterator<IndexType>(n),
									result_ptr,
									pred);

}


__BEGIN_DECLS__

index_type CountValueI(const index_type* data, index_type size, index_type value) {
	thrust::device_ptr<const index_type> data_ptr(data);
	return thrust::count(data_ptr, data_ptr + size, value);
}


void FindValueI(const index_type* data, index_type size, index_type value, index_type* result) {
	FindValue<index_type, index_type>(data, size, value, result);
}	

index_type CountValueColorLegacy(const color_t* data, index_type size, color_t value) {
	thrust::device_ptr<const color_t> data_ptr(data);
	return thrust::count(data_ptr, data_ptr + size, value);
}
index_type CountValueColor(const color_t* data, index_type size, color_t value, void* buffer) {
	return CountValue<index_type, color_t>(data, size, value, buffer);
}

void FindValueColor(const color_t* data, index_type size, color_t value, index_type* result) {
	FindValue<index_type, color_t>(data, size, value, result);
}

__END_DECLS__
