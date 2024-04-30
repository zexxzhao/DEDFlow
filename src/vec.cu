#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include "vec.h"


__BEGIN_DECLS__

void VecAXPY(value_type a, const value_type* x, value_type* y, index_type n) {
	thrust::device_ptr<const value_type> x_ptr(x);
	thrust::device_ptr<value_type> y_ptr(y);
	thrust::transform(x_ptr, x_ptr + n, y_ptr, y_ptr, thrust::placeholders::_1 * a + thrust::placeholders::_2);
}

void VecPointwiseMult(const value_type* a, const value_type* b, value_type* c, index_type n) {
	thrust::device_ptr<const value_type> a_ptr(a);
	thrust::device_ptr<const value_type> b_ptr(b);
	thrust::device_ptr<value_type> c_ptr(c);
	thrust::transform(a_ptr, a_ptr + n, b_ptr, c_ptr, thrust::multiplies<value_type>());
}

void VecPointwiseDiv(const value_type* a, const value_type* b, value_type* c, index_type n) {
	thrust::device_ptr<const value_type> a_ptr(a);
	thrust::device_ptr<const value_type> b_ptr(b);
	thrust::device_ptr<value_type> c_ptr(c);
	thrust::transform(a_ptr, a_ptr + n, b_ptr, c_ptr, thrust::divides<value_type>());
}

__END_DECLS__
