#ifndef __INDEXING_H__
#define __INDEXING_H__

#include "common.h"
#include "alloc.h"
#include "color.h"

__BEGIN_DECLS__

index_type CountValueI(const index_type* data, index_type n, index_type value, Arena scratch);
index_type CountValueImp(const void* data, index_type elem_size, index_type n, void* value, Arena scratch);

void SortByKeyI(index_type* keys, index_type* values, index_type n, Arena scratch);


__END_DECLS__

#endif /* __INDEXING_H__ */
