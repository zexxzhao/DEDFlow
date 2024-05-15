#ifndef __INDEXING_H__
#define __INDEXING_H__

#include "common.h"
#include "color.h"

__BEGIN_DECLS__

index_type CountValueI(const index_type* data, index_type n, index_type value);
void FindValueI(const index_type* data, index_type n, index_type value, index_type* result);

index_type CountValueColor(const color_t* data, index_type n, color_t value);
void FindValueColor(const color_t* data, index_type n, color_t value, index_type* result);

__END_DECLS__

#endif /* __INDEXING_H__ */
