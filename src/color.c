#include <string.h>
#include "vec.h"
#include "Mesh.h"
#include "csr.h"
#include "color.h"
#include "color_impl.h"



__BEGIN_DECLS__

void ColorMeshTet(CdamMesh* mesh, index_type max_color_len, index_type* color, Arena scratch) {

	index_type* ien = CdamMeshTet(mesh);
	index_type num_elem = CdamMeshNumTet(mesh);
	index_type num_node = CdamMeshNumNode(mesh);
	// u32 color_count[MAX_COLOR];
	index_type* v2e_row_ptr, *v2e_col_ind;
	index_type v2e_nnz;

	// memset(color, UNCOLORED, num_elem * sizeof(index_type));
	CdamMemset(color, UNCOLORED, num_elem * sizeof(index_type), DEVICE_MEM);
	// CdamMemset(color_count, 0, MAX_COLOR * sizeof(u32), HOST_MEM);

	
	/* Build a vertex2element graph */	
	v2e_row_ptr = (index_type*)ArenaPush(sizeof(index_type), num_node + 1, &scratch, 0);
	GenerateV2EMapRowTetGPU(ien, num_elem, num_node, v2e_row_ptr);
	CdamMemcpy(&v2e_nnz, &v2e_row_ptr + num_node, sizeof(index_type), HOST_MEM, DEVICE_MEM);
	v2e_col_ind = (index_type*)ArenaPush(sizeof(index_type), v2e_nnz, &scratch, 0);
	GenerateV2EMapColTetGPU(ien, num_elem, num_node, v2e_row_ptr, v2e_col_ind);

	/* Jones-Plassman-Luby algorithm */
	ColorElementJPLTetGPU(ien, v2e_row_ptr, v2e_col_ind, MAX_COLOR, color, num_elem);
}


void ColorMeshPrism(CdamMesh* mesh, index_type max_color_len, index_type* color, Arena scratch) {}
void ColorMeshHex(CdamMesh* mesh, index_type max_color_len, index_type* color, Arena scratch) {}

index_type GetMaxColor(const index_type* color, index_type size) {
	index_type max_color = 0;
	GetMaxColorGPU(color, size, &max_color);
	return max_color;
}

__END_DECLS__
