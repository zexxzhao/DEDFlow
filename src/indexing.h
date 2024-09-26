#ifndef __INDEXING_H__
#define __INDEXING_H__

#include "common.h"
#include "alloc.h"
#include "color.h"


__BEGIN_DECLS__

index_type CountValueI(const index_type* data, index_type n, index_type value, Arena scratch);
index_type CountValueImp(const void* data, index_type elem_size, index_type n, void* value, Arena scratch);

void SortByKeyI(index_type* keys, index_type* values, index_type n, Arena scratch);


index_type CountIndexByRangeGPU(index_type n, index_type* index, index_type start, index_type end);
void Iota(index_type* data, index_type n, index_type start, index_type step);

void GenMaskByRange(index_type nrow, index_type ncol, index_type* input,
										byte* mask, index_type start, index_type end, Arena scratch); 
__END_DECLS__


#endif /* __INDEXING_H__ */
