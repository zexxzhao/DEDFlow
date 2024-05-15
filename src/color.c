#include <cublas_v2.h>
#include <string.h>
#include "alloc.h"
#include "vec.h"
#include "Mesh.h"
#include "csr.h"
#include "color.h"
#include "color_impl.h"



__BEGIN_DECLS__

void ColorMeshTet(const Mesh3D* mesh, index_type max_color_len, color_t* color) {
	const Mesh3DData* device_data = Mesh3DDevice(mesh);

	const index_type* ien = Mesh3DDataTet(device_data);
	index_type num_elem = Mesh3DDataNumTet(device_data);
	index_type num_node = Mesh3DDataNumNode(device_data);
	u32 color_count[MAX_COLOR];

	// memset(color, UNCOLORED, num_elem * sizeof(index_type));
	cudaMemset(color, UNCOLORED, num_elem * sizeof(index_type));
	memset(color_count, 0, MAX_COLOR * sizeof(u32));
	
	/* Build a vertex2element graph */	
	index_type* v2e_row_ptr = (index_type*)CdamMallocDevice((num_node + 1) * sizeof(index_type));
	GenerateV2EMapRowTetGPU(ien, num_elem, num_node, v2e_row_ptr);
	index_type v2e_nnz;
	cublasGetVector(1, sizeof(index_type), v2e_row_ptr + num_node, 1, &v2e_nnz, 1);
	index_type* v2e_col_ind = (index_type*)CdamMallocDevice(v2e_nnz * sizeof(index_type));
	GenerateV2EMapColTetGPU(ien, num_elem, num_node, v2e_row_ptr, v2e_col_ind);

	
	/* Jones-Plassman-Luby algorithm */
	ColorElementJPLTetGPU(ien, v2e_row_ptr, v2e_col_ind, MAX_COLOR, color, num_elem);

}


void ColorMeshPrism(const Mesh3D* mesh, index_type max_color_len, color_t* color) {}
void ColorMeshHex(const Mesh3D* mesh, index_type max_color_len, color_t* color) {}

color_t GetMaxColor(const color_t* color, index_type size) {
	color_t max_color = 0;
	GetMaxColorGPU(color, size, &max_color);
	return max_color;
}

__END_DECLS__
