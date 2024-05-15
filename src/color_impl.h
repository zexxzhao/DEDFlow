#ifndef __COLOR_IMPL_H__
#define __COLOR_IMPL_H__

#include "common.h"
#include "color.h"

__BEGIN_DECLS__

void GenerateV2EMapRowTetGPU(const index_type* ien, index_type num_elem, index_type num_node, index_type* row_ptr);
void GenerateV2EMapColTetGPU(const index_type* ien, index_type num_elem, index_type num_node, const index_type* row_ptr, index_type* col_idx);

void GenerateV2EMapRowPrismGPU(const index_type* ien, index_type num_elem, index_type num_node, index_type* row_ptr);
void GenerateV2EMapColPrismGPU(const index_type* ien, index_type num_elem, index_type num_node, const index_type* row_ptr, index_type* col_idx);

void GenerateV2EMapRowHexGPU(const index_type* ien, index_type num_elem, index_type num_node, index_type* row_ptr);
void GenerateV2EMapColHexGPU(const index_type* ien, index_type num_elem, index_type num_node, const index_type* row_ptr, index_type* col_idx);

void ColorElementJPLTetGPU(const index_type* ien, const index_type* row_ptr, const index_type* col_ind,
													 color_t max_color, color_t* color, index_type num_elem);


void GenerateRandomColor(color_t* color, index_type num_elem, color_t max_color);

void GetMaxColorGPU(const color_t* color, index_type num_elem, color_t* max_color);
__END_DECLS__
#endif /* __COLOR_IMPL_H__ */
