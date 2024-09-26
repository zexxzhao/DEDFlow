#include "vec_impl.h"

#ifdef CDAM_USE_CUDA
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
template<typename T>
struct VecModFunctor {
	T b;
	VecModFunctor(T b) : b(b) {}
	__host__ __device__ T operator()(T x) const {
		return x - b * floor(x / b);
	}
};

template<typename I, typename T>
__global__ void
VecPointwiseMultKernel(const T* a, const T* b, T* c, I n) {
	I i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] * b[i];
	}
}

template<typename I, typename T>
__global__ void
VecPointwiseDivKernel(const T* a, const T* b, T* c, I n) {
	I i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] / b[i];
	}
}

template<typename I, typename T>
__global__ void
VecPointwiseInvKernel(T* a, I n) {
	I i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		a[i] = 1.0 / a[i];
	}
}

__BEGIN_DECLS__

void VecAXPY(value_type a, const value_type* x, value_type* y, index_type n) {
	thrust::device_ptr<const value_type> x_ptr(x);
	thrust::device_ptr<value_type> y_ptr(y);
	thrust::transform(x_ptr, x_ptr + n, y_ptr, y_ptr, thrust::placeholders::_1 * a + thrust::placeholders::_2);
}

void VecPointwiseMult(const value_type* a, const value_type* b, value_type* c, index_type n) {
	int block_dim = 256;
	int grid_dim = CEIL_DIV(n, block_dim);
	VecPointwiseMultKernel<<<grid_dim, block_dim>>>(a, b, c, n);
}

void VecPointwiseDiv(const value_type* a, const value_type* b, value_type* c, index_type n) {
	// thrust::device_ptr<const value_type> a_ptr(a);
	// thrust::device_ptr<const value_type> b_ptr(b);
	// thrust::device_ptr<value_type> c_ptr(c);
	// thrust::transform(a_ptr, a_ptr + n, b_ptr, c_ptr, thrust::divides<value_type>());
	int block_dim = 256;
	int grid_dim = (n + block_dim - 1) / block_dim;
	VecPointwiseDivKernel<<<grid_dim, block_dim>>>(a, b, c, n);
}

void VecPointwiseInv(value_type* a, index_type n) {
	int block_dim = 256;
	int grid_dim = (n + block_dim - 1) / block_dim;
	VecPointwiseInvKernel<<<grid_dim, block_dim>>>(a, n);
}


void VecPointwiseMod(const value_type* a, value_type b, value_type* c, index_type n) {
	thrust::device_ptr<const value_type> a_ptr(a);
	thrust::device_ptr<value_type> c_ptr(c);
	thrust::transform(a_ptr, a_ptr + n, c_ptr, VecModFunctor<value_type>(b));
}


__END_DECLS__
#else
__BEGIN_DECLS__
void VecPointwiseMult(const value_type* a, const value_type* b, value_type* c, index_type n) {
	for (index_type i = 0; i < n; i++) {
		c[i] = a[i] * b[i];
	}
}

void VecPointwiseDiv(const value_type* a, const value_type* b, value_type* c, index_type n) {
	for (index_type i = 0; i < n; i++) {
		c[i] = a[i] / b[i];
	}
}

void VecPointwiseInv(value_type* a, index_type n) {
	for (index_type i = 0; i < n; i++) {
		a[i] = 1.0 / a[i];
	}
}
__END_DECLS__
#endif
