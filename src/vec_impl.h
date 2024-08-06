#ifndef __VEC_IMPL_H__
#define __VEC_IMPL_H__

#include "common.h"

__BEGIN_DECLS__
void VecAXPY(value_type a, const value_type* x, value_type* y, index_type n);
void VecPointwiseMult(const value_type* x, const value_type* y, value_type* z, index_type n);
void VecPointwiseDiv(const value_type* x, const value_type* y, value_type* z, index_type n);
void VecPointwiseInv(value_type* x, index_type n);

__END_DECLS__

#endif /* __VEC_IMPL_H__ */
