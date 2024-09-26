#include <string.h>
#include "layout.h"
#include "Mesh.h"
#include "csr.h"
#include "color.h"
#include "color_impl.h"



__BEGIN_DECLS__
static void GenerateRandomColorHost(index_type* color, index_type num_elem, index_type max_color) {
	index_type i;
	for(i = 0; i < num_elem; i++) {
		color[i] = rand() % (COLOR_RANDOM_UB - COLOR_RANDOM_LB) + COLOR_RANDOM_LB;
	}
}
static void ColorElementJPLStep(const index_type* ien, /* Element to Vertex */
																const index_type* row_ptr, const index_type* col_ind, /* Vertex to Element */
																index_type max_color, index_type* color, index_type num_elem, int NSHL) {
	b32 found_max;
	for(index_type i = 0; i < num_elem; i++) {
		if(color[i] < 0) continue;
		found_max = TRUE;
		for(int j = 0; j < NSHL; ++j) {
			index_type node = ien[i * NSHL + j];
			index_type start = row_ptr[node];
			index_type end = row_ptr[node + 1];
			index_type k, elem;

			for (k = start; k < end; ++k) {
				elem = col_ind[k];
				if (color[i] < color[elem] || (color[i] == color[elem] && elem > i)) {
					found_max = FALSE;
					break;
				}
			}
		}
		if(found_max) color[i] = COLOR_MARKED;
	}
}

static void ColorElementJPLHost(const index_type* ien, /* Element to Vertex */
																const index_type* row_ptr, const index_type* col_ind, /* Vertex to Element */
																index_type max_color, index_type* color, index_type num_elem, int NSHL) {
	index_type left = num_elem;
	index_type c;

	GenerateRandomColorHost(color, num_elem, max_color);


	for(c = 0; c < max_color && left; c++) {
		index_type reverse_color = -1 - c;
		/* Find the uncolored local maximum */
		ColorElementJPLStep(ien, row_ptr, col_ind, max_color, color, num_elem, NSHL);
		/* Reverse the color of the local maximum */
		for(index_type i = 0; i < num_elem; i++) {
			if(color[i] == COLOR_MARKED) {
				color[i] = reverse_color;
				left--;
			}
		}
	}
	/* Recover the original color */
	for(index_type i = 0; i < num_elem; i++) {
		color[i] = -1 - color[i];
	}
}

void ColorElementJPL(const index_type* ien, /* Element to Vertex */
											const index_type* row_ptr, const index_type* col_ind, /* Vertex to Element */
											index_type max_color, index_type* color, index_type num_elem, int NSHL) {
#ifdef CDAM_USE_CUDA
	ColorElementJPLDevice(ien, row_ptr, col_ind, max_color, color, num_elem, NSHL);
#else
	ColorElementJPLHost(ien, row_ptr, col_ind, max_color, color, num_elem, NSHL);
#endif
}

static void GenerateV2EMapRowHost(const index_type* ien, index_type num_elem, index_type num_node, index_type* v2e_row_ptr, int NSHL) {
	index_type i, j;
	CdamMemset(v2e_row_ptr, 0, sizeof(index_type) * (num_node + 1), HOST_MEM);
	for(i = 0; i < num_elem; i++) {
		for(j = 0; j < NSHL; j++) {
			v2e_row_ptr[ien[i * NSHL + j] + 1]++;
		}
	}
	v2e_row_ptr[num_node] = 0;
	for(i = 0; i < num_node; i++) {
		v2e_row_ptr[i + 1] += v2e_row_ptr[i];
	}
}

static void GenerateV2EMapRow(const index_type* ien, index_type num_elem, index_type num_node, index_type* v2e_row_ptr, int NSHL) {
#ifdef CDAM_USE_CUDA
	GenerateV2EMapRowDevice(ien, num_elem, num_node, v2e_row_ptr, NSHL);
#else
	GenerateV2EMapRowHost(ien, num_elem, num_node, v2e_row_ptr, NSHL);
#endif

}

static void GenerateV2EMapColHost(const index_type* ien, index_type num_elem, index_type num_node, const index_type* v2e_row_ptr, index_type* v2e_col_ind, int NSHL) {
	index_type i, j;
	index_type* tmp_row_ptr = CdamTMalloc(index_type, num_node + 1, HOST_MEM);
	CdamMemcpy(tmp_row_ptr, v2e_row_ptr, sizeof(index_type) * (num_node + 1), HOST_MEM, HOST_MEM);
	for(i = 0; i < num_elem; i++) {
		for(j = 0; j < NSHL; j++) {
			index_type node = ien[i * NSHL + j];
			v2e_col_ind[tmp_row_ptr[node]++] = i;
		}
	}
	CdamFree(tmp_row_ptr, sizeof(index_type) * (num_node + 1), HOST_MEM);
}

void GenerateV2EMapCol(const index_type* ien, index_type num_elem, index_type num_node, const index_type* v2e_row_ptr, index_type* v2e_col_ind, int NSHL) {
#ifdef CDAM_USE_CUDA
	GenerateV2EMapColDevice(ien, num_elem, num_node, v2e_row_ptr, v2e_col_ind, NSHL);
#else
	GenerateV2EMapColHost(ien, num_elem, num_node, v2e_row_ptr, v2e_col_ind, NSHL);
#endif
}

void ColorMeshElement(index_type* ien, index_type num_elem, index_type NSHL,
											index_type num_node, index_type max_color_len,
											index_type* color) {
	index_type* v2e_row_ptr, *v2e_col_ind;
	index_type v2e_nnz;

	/* Build a vertex2element graph */
	v2e_row_ptr = CdamTMalloc(index_type, num_node + 1, DEVICE_MEM);
	GenerateV2EMapRow(ien, num_elem, num_node, v2e_row_ptr, NSHL);

	v2e_nnz = v2e_row_ptr[num_node];
	CdamMemcpy(&v2e_nnz, &v2e_row_ptr + num_node, sizeof(index_type), HOST_MEM, DEVICE_MEM);

	v2e_col_ind = CdamTMalloc(index_type, v2e_nnz, DEVICE_MEM);
	GenerateV2EMapCol(ien, num_elem, num_node, v2e_row_ptr, v2e_col_ind, NSHL);

	/* Jones-Plassman-Luby algorithm */
	ColorElementJPL(ien, v2e_row_ptr, v2e_col_ind, max_color_len, color, num_elem, NSHL);

	CdamFree(v2e_row_ptr, sizeof(index_type) * (num_node + 1), DEVICE_MEM);
	CdamFree(v2e_col_ind, sizeof(index_type) * v2e_nnz, DEVICE_MEM);
}

void ColorMeshTet(CdamMesh* mesh, index_type max_color_len, index_type* color, Arena scratch) {

	index_type* ien = CdamMeshIEN(mesh);
	index_type num_tet = CdamMeshNumTet(mesh);
	index_type num_node = CdamMeshNumNode(mesh);
	index_type NSHL = 4;

	ColorMeshElement(ien, num_tet, NSHL, num_node, max_color_len, color);
}


void ColorMeshPrism(CdamMesh* mesh, index_type max_color_len, index_type* color, Arena scratch) {
	index_type* ien = CdamMeshIEN(mesh);
	index_type num_tet = CdamMeshNumTet(mesh);
	index_type num_prism = CdamMeshNumPrism(mesh);
	index_type num_node = CdamMeshNumNode(mesh);
	index_type NSHL = 6;

	ien += num_tet * 4;
	ColorMeshElement(ien, num_prism, NSHL, num_node, max_color_len, color + num_tet);
}
void ColorMeshHex(CdamMesh* mesh, index_type max_color_len, index_type* color, Arena scratch) {
	index_type* ien = CdamMeshIEN(mesh);
	index_type num_tet = CdamMeshNumTet(mesh);
	index_type num_prism = CdamMeshNumPrism(mesh);
	index_type num_hex = CdamMeshNumHex(mesh);
	index_type num_node = CdamMeshNumNode(mesh);
	index_type NSHL = 8;

	ien += (num_tet * 4 + num_prism * 6);
	ColorMeshElement(ien, num_hex, NSHL, num_node, max_color_len, color + num_tet + num_prism);
}

index_type GetMaxColor(const index_type* color, index_type size, Arena scratch) {
	index_type max_color = 0;
#ifdef CDAM_USE_CUDA
	GetMaxColorGPU(color, size, &max_color, scratch);
#else
	for(index_type i = 0; i < size; i++) {
		if(color[i] > max_color) max_color = color[i];
	}
#endif
	return max_color;
}

__END_DECLS__
