
#include <string.h>
#include <stdio.h>

#include "alloc.h"
#include "Mesh.h"

#include "csr.h"
#include "csr_impl.h"
#define PREALLOC_SIZE 64
__BEGIN_DECLS__
typedef struct CSRHashMap CSRHashMap;
struct CSRHashMap {
	csr_index_type* buff;
	csr_index_type* row_len;
	u32 size;
};
static CSRHashMap*
CSRHashMapCreate(u32 size) {
	CSRHashMap* map = (CSRHashMap*)CdamMallocHost(sizeof(CSRHashMap));
	map->buff = (u32*)CdamMallocHost(sizeof(u32) * size * PREALLOC_SIZE);
	memset(map->buff, 0, sizeof(u32) * size * PREALLOC_SIZE);
	map->row_len = (u32*)CdamMallocHost(sizeof(u32) * size);
	memset(map->row_len, 0, sizeof(u32) * size);
	map->size = size;
	return map;
}

static void
CSRHashMapDestroy(CSRHashMap* map) {
	CdamFreeHost(map->buff, map->size * PREALLOC_SIZE * sizeof(csr_index_type));
	CdamFreeHost(map->row_len, map->size * sizeof(csr_index_type));
	CdamFreeHost(map, sizeof(map));
}

static csr_index_type* lower_bound(csr_index_type* first,
																	 csr_index_type* last,
																	 csr_index_type value) {
	csr_index_type* it;
	csr_index_type count, step;
	count = (csr_index_type)(last - first);
	while(count > 0) {
		it = first;
		step = count / 2;
		it += step;
		if(*it < value) {
			first = ++it;
			count -= step + 1;
		} else {

			count = step;
		}
	}
	return first;
}

static u32 CSRHashMapPush(CSRHashMap* map, u32 key, u32 value) {
	u32* buff = map->buff;
	u32* row_len = map->row_len;
	u32* row = buff + key * PREALLOC_SIZE;

	u32 len = row_len[key];
	ASSERT(len < PREALLOC_SIZE && "CSRHashMapPush: row overflow");

	u32* it = lower_bound(row, row + len, value);
	if (it == row + len) { /* the value is the larger than the existing ones */
		*it = value;
		row_len[key]++;
	}
	else if (*it != value) { /* the value is not in the row but in the middle */
		memmove(it + 1, it, (row + len - it) * sizeof(u32));
		*it = value;
		row_len[key]++;
	}
	else { /* the value is already in the row */
	}

	return row_len[key];
}

static CSRHashMap*
GetNodalGraphFromMesh(const Mesh3D* mesh) {
	Mesh3DData* host = Mesh3DHost(mesh);
	u32 num_node = Mesh3DNumNode(host);
	u32 num_tet = Mesh3DNumTet(host);
	u32 num_prism = Mesh3DNumPrism(host);
	u32 num_hex = Mesh3DNumHex(host);
	u32* ien_tet = Mesh3DDataTet(host);
	u32* ien_prism = Mesh3DDataPrism(host);
	u32* ien_hex = Mesh3DDataHex(host);
	
	u32 i, j, k;
	u32* elem;
	CSRHashMap* map = CSRHashMapCreate(num_node);
	/* tet */
	for(k = 0; k < num_tet; k++) {
		elem = ien_tet + k * 4;
		for(i = 0; i < 4; i++) {
			CSRHashMapPush(map, elem[i], elem[i]);
			for(j = 0; j < 4; j++) {
				if(i != j) {
					CSRHashMapPush(map, elem[i], elem[j]);
				}
			}
		}
	}
	/* prism */
	for(k = 0; k < num_prism; k++) {
		elem = ien_prism + k * 6;
		for(i = 0; i < 6; i++) {
			CSRHashMapPush(map, elem[i], elem[i]);
			for(j = 0; j < 6; j++) {
				if(i != j) {
					CSRHashMapPush(map, elem[i], elem[j]);
				}
			}
		}
	}
	/* hex */
	for(k = 0; k < num_hex; k++) {
		elem = ien_hex + k * 8;
		for(i = 0; i < 8; i++) {
			CSRHashMapPush(map, elem[i], elem[i]);
			for(j = 0; j < 8; j++) {
				if(i != j) {
					CSRHashMapPush(map, elem[i], elem[j]);
				}
			}
		}
	}

	return map;
}

#include "csr_impl.h"

// CSRAttr* CSRAttrCreate(const Mesh3D* mesh) {
// 	CSRAttr* attr = (CSRAttr*)CdamMallocHost(sizeof(CSRAttr));
// 
// 	GenerateCSRFromMesh(mesh, attr);
// }

CSRAttr* CSRAttrCreate(const Mesh3D* mesh) {
	CSRAttr* attr = (CSRAttr*)CdamMallocHost(sizeof(CSRAttr));
	u32 num_node = Mesh3DNumNode(mesh);
	u32 i, j;
	u32 nnz = 0;
	CSRAttrNumRow(attr) = num_node;
	CSRAttrNumCol(attr) = num_node;

	CSRHashMap* map = GetNodalGraphFromMesh(mesh);
	/* count nnz */
	for(i = 0; i < num_node; i++) {
		nnz += map->row_len[i];
	}
	CSRAttrNNZ(attr) = nnz;
	csr_index_type* row_ptr= (csr_index_type*)CdamMallocHost(sizeof(csr_index_type) * (num_node + 1));
	csr_index_type* col_ind= (csr_index_type*)CdamMallocHost(sizeof(csr_index_type) * nnz);

	row_ptr[0] = 0;
	for (i = 0; i < num_node; i++) {
		row_ptr[i + 1] = row_ptr[i] + map->row_len[i];
		memcpy(col_ind + row_ptr[i], map->buff + i * PREALLOC_SIZE, sizeof(csr_index_type) * map->row_len[i]);
	}

	if(0){
		FILE* fp = fopen("csr.txt", "w");
		for(i = 0; i < num_node; i++) {
			fprintf(fp, "%d: ", i);
			for(j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
				fprintf(fp, "%d ", col_ind[j]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	CSRHashMapDestroy(map);

	csr_index_type* row_ptr_dev = (csr_index_type*)CdamMallocDevice(sizeof(csr_index_type) * (num_node + 1));
	csr_index_type* col_ind_dev = (csr_index_type*)CdamMallocDevice(sizeof(csr_index_type) * nnz);
	cudaMemcpy(row_ptr_dev, row_ptr, sizeof(csr_index_type) * (num_node + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(col_ind_dev, col_ind, sizeof(csr_index_type) * nnz, cudaMemcpyHostToDevice);
	CdamFreeHost(row_ptr, sizeof(csr_index_type) * (num_node + 1));
	CdamFreeHost(col_ind, sizeof(csr_index_type) * nnz);
	CSRAttrRowPtr(attr) = row_ptr_dev;
	CSRAttrColInd(attr) = col_ind_dev;
	return attr;
}


CSRAttr* CSRAttrCreateBlock(const CSRAttr* attr, csr_index_type block_row, csr_index_type block_col) {
	CSRAttr* new_attr = (CSRAttr*)CdamMallocHost(sizeof(CSRAttr));
	u32 nnz = CSRAttrNNZ(attr);
	u32 num_row = CSRAttrNumRow(attr);
	u32 num_col = CSRAttrNumCol(attr);

	CSRAttrNumRow(new_attr) = num_row * block_row;
	CSRAttrNumCol(new_attr) = num_col * block_col;

	CSRAttrNNZ(new_attr) = nnz * block_row * block_col;

	CSRAttrRowPtr(new_attr) = (csr_index_type*)CdamMallocDevice(sizeof(csr_index_type) * (num_row * block_row + 1));
	CSRAttrColInd(new_attr) = (csr_index_type*)CdamMallocDevice(sizeof(csr_index_type) * nnz * block_row * block_col);

	csr_index_type block_size[2] = {block_row, block_col};
	ExpandCSRByBlockSize(attr, new_attr, block_size);
	return new_attr;
}

void CSRAttrDestroy(CSRAttr* attr) {
	CdamFreeDevice(CSRAttrRowPtr(attr), (CSRAttrNumRow(attr) + 1) * sizeof(csr_index_type));
	CdamFreeDevice(CSRAttrColInd(attr), CSRAttrNNZ(attr) * sizeof(csr_index_type));
	CdamFreeHost(attr, sizeof(CSRAttr));
}

u32 CSRAttrLength(CSRAttr *attr, csr_index_type row) {
	return CSRAttrRowPtr(attr)[row + 1] - CSRAttrRowPtr(attr)[row];
}

csr_index_type* CSRAttrRow(CSRAttr *attr, csr_index_type row) {
	return CSRAttrColInd(attr) + CSRAttrRowPtr(attr)[row];
}

void CSRAttrGetNonzeroIndBatched(const CSRAttr* attr, csr_index_type batch_size,
																 const index_type* row, const index_type* col,
																 csr_index_type* ind) {
	CSRAttrGetNZIndBatchedGPU(attr, batch_size, row, col, ind);
}

__END_DECLS__

