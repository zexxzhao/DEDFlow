#ifndef __COLOR_IMPL_H__
#define __COLOR_IMPL_H__

#include "common.h"
#include "color.h"

#ifdef CDAM_USE_CUDA

__BEGIN_DECLS__

void GenerateV2EMapRowDevice(const index_type* ien, index_type num_elem, index_type nshl, index_type num_node, index_type* row_ptr);
void GenerateV2EMapColDevice(const index_type* ien, index_type num_elem, index_type nshl, index_type num_node, const index_type* row_ptr, index_type* col_idx);


void ColorElementJPLDevice(const index_type* ien,
													 const index_type* row_ptr, const index_type* col_ind,
													 index_type max_color, index_type* color, index_type num_elem,
													 int NSHL);


void GenerateRandomColorDevice(index_type* color, index_type num_elem, index_type max_color);

void GetMaxColorDevice(const index_type* color, index_type num_elem, index_type* max_color);

__END_DECLS__

#endif /* CDAM_USE_CUDA */
#endif /* __COLOR_IMPL_H__ */
