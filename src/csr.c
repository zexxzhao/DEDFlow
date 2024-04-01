
#include <string.h>
#include "alloc.h"
#include "Mesh.h"

#include "csr.h"

#define PREALLOC_SIZE 64
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
	if (it == row + len) { /* value is the largest */
		*it = value;
		row_len[key]++;
	}
	else if (*it != value) { /* value is not in the row */
		memmove(it + 1, it, (row + len - it) * sizeof(u32));
		*it = value;
		row_len[key]++;
	}
	else { /* else value is already in the row */
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
			for(j = 0; j < 8; j++) {
				if(i != j) {
					CSRHashMapPush(map, elem[i], elem[j]);
				}
			}
		}
	}

	return map;
}

CSRAttr* CSRAttrCreate(const Mesh3D* mesh) {
	CSRAttr* attr = (CSRAttr*)CdamMallocHost(sizeof(CSRAttr));
	u32 num_node = Mesh3DNumNode(mesh);
	u32 i;
	u32 nnz = 0;
	CSRAttrNumRow(attr) = num_node;	
	CSRAttrNumCol(attr) = num_node;

	CSRHashMap* map = GetNodalGraphFromMesh(mesh);
	/* count nnz */
	for(i = 0; i < num_node; i++) {
		nnz += map->row_len[i];
	}
	CSRAttrNNZ(attr) = nnz;
	CSRAttrRowPtr(attr) = (csr_index_type*)CdamMallocHost(sizeof(csr_index_type) * (num_node + 1));
	CSRAttrColInd(attr) = (csr_index_type*)CdamMallocHost(sizeof(csr_index_type) * nnz);

	csr_index_type* row_ptr = CSRAttrRowPtr(attr);
	csr_index_type* col_ind = CSRAttrColInd(attr);

	row_ptr[0] = 0;
	for(i = 0; i < num_node; i++) {
		row_ptr[i + 1] = row_ptr[i] + map->row_len[i];
		memcpy(col_ind + row_ptr[i], map->buff + i * PREALLOC_SIZE, sizeof(u32) * map->row_len[i]);
	}
	CSRHashMapDestroy(map);
	return attr;
}

void CSRAttrDestroy(CSRAttr* attr) {
	CdamFreeHost(CSRAttrRowPtr(attr), (CSRAttrNumRow(attr) + 1) * sizeof(csr_index_type));
	CdamFreeHost(CSRAttrColInd(attr), CSRAttrNNZ(attr) * sizeof(csr_index_type));
	CdamFreeHost(attr, sizeof(CSRAttr));
}

u32 CSRAttrLength(const CSRAttr *attr, csr_index_type row) {
	return CSRAttrRowPtr(attr)[row + 1] - CSRAttrRowPtr(attr)[row];
}

csr_index_type* CSRAttrRow(const CSRAttr *attr, csr_index_type row) {
	return CSRAttrColInd(attr) + CSRAttrRowPtr(attr)[row];
}


CSRMatrix* CSRMatrixCreate(CSRAttr* attr, csr_index_type block_size) {
	CSRMatrix* mat = (CSRMatrix*)CdamMallocHost(sizeof(CSRMatrix));
	CSRMatrixAttr(mat) = attr;
	CSRMatrixBS(mat) = block_size;
	CSRMatrixData(mat) = (csr_value_type*)CdamMallocHost(sizeof(csr_value_type) * CSRAttrNNZ(attr) * block_size * block_size);
	ASSERT(block_size >= 1 && "CSRMatrixCreate: block_size must be greater than 1.");
	if(block_size == 1) {
		cusparseCreateCsr(&CSRMatrixDescr(mat),
											(i64)CSRAttrNumRow(attr), (i64)CSRAttrNumCol(attr), (i64)CSRAttrNNZ(attr),
											CSRAttrRowPtr(attr), CSRAttrColInd(attr), CSRMatrixData(mat),
											CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	}
	else {
		cusparseCreateBsr(&CSRMatrixDescr(mat),
											CSRAttrNumRow(attr), CSRAttrNumCol(attr),
											CSRAttrNNZ(attr), block_size, block_size, 
											CSRAttrRowPtr(attr), CSRAttrColInd(attr),
											CSRMatrixData(mat),
											CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F,
											CUSPARSE_ORDER_ROW);
	}
	return mat;
}
