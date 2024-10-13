#ifndef __INDEXING_H__
#define __INDEXING_H__

#include "common.h"
#include "alloc.h"
#include "color.h"


__BEGIN_DECLS__

index_type CountValueI(index_type* data, index_type n, index_type value, Arena scratch);
index_type CountValueImp(void* data, index_type elem_size, index_type n, void* value, Arena scratch);

void SortByKeyI(index_type* keys, index_type* values, index_type n, Arena scratch);


index_type CountIndexByRangeGPU(index_type n, index_type* index, index_type start, index_type end);
void Iota(index_type* data, index_type n, index_type start, index_type step);

void GenMaskByRange(index_type nrow, index_type ncol, index_type* input,
										byte* mask, index_type start, index_type end, Arena scratch); 

void RestrictVec(void* dst, void* src, index_type n, index_type* index, size_t elem_size);
void ProlongateVec(void* dst, void* src, index_type n, index_type* index, size_t elem_size);

void RestrictAddVecStrided(value_type* dst, index_type dst_stride,
													 value_type* src, index_type src_stride,
													 index_type n, index_type* index, index_type block_size); 

void ProlongateAddVecStrided(value_type* dst, index_type dst_stride,
														 value_type* src, index_type src_stride,
														 index_type n, index_type* index, index_type block_size);

__END_DECLS__


#endif /* __INDEXING_H__ */
