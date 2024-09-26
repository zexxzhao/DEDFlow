
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
	index_type* buff;
	index_type* row_len;
	index_type size;
};
static CSRHashMap*
CSRHashMapCreate(index_type size) {
	CSRHashMap* map = CdamTMalloc(CSRHashMap, 1, HOST_MEM);
	map->buff = CdamTMalloc(index_type, size * PREALLOC_SIZE, HOST_MEM);
	CdamMemset(map->buff, 0, sizeof(index_type) * size * PREALLOC_SIZE, HOST_MEM);
	map->row_len = CdamTMalloc(index_type, size, HOST_MEM);
	CdamMemset(map->row_len, 0, sizeof(index_type) * size, HOST_MEM);
	map->size = size;
	return map;
}

static void
CSRHashMapDestroy(CSRHashMap* map) {
	CdamFree(map->buff, map->size * PREALLOC_SIZE * sizeof(index_type), HOST_MEM);
	CdamFree(map->row_len, map->size * sizeof(index_type), HOST_MEM);
	CdamFree(map, sizeof(CSRHashMap), HOST_MEM);
}

static index_type* lower_bound(index_type* first,
																	 index_type* last,
																	 index_type value) {
	index_type* it;
	index_type count, step;
	count = (index_type)(last - first);
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

static index_type CSRHashMapPush(CSRHashMap* map, index_type key, index_type value) {
	index_type* buff = map->buff;
	index_type* row_len = map->row_len;
	index_type* row = buff + key * PREALLOC_SIZE;

	index_type len = row_len[key];
	ASSERT(len < PREALLOC_SIZE && "CSRHashMapPush: row overflow");

	index_type* it = lower_bound(row, row + len, value);
	if (it == row + len) { /* the value is the larger than the existing ones */
		*it = value;
		row_len[key]++;
	}
	else if (*it != value) { /* the value is not in the row but in the middle */
		memmove(it + 1, it, (row + len - it) * sizeof(index_type));
		*it = value;
		row_len[key]++;
	}
	else { /* the value is already in the row */
	}

	return row_len[key];
}

static CSRHashMap*
GetNodalGraphFromMesh(CdamMesh* mesh) {

	index_type num_node = CdamMeshNumNode(mesh);
	index_type num_tet = CdamMeshNumTet(mesh);
	index_type num_prism = CdamMeshNumPrism(mesh);
	index_type num_hex = CdamMeshNumHex(mesh);
	index_type* ien_tet = CdamMeshTet(mesh);
	index_type* ien_prism = CdamMeshPrism(mesh);
	index_type* ien_hex = CdamMeshHex(mesh);
	
	index_type i, j, k;
	index_type* elem;
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

static void ExpandCSRByBlockSizeHost(const CSRAttr* csr, CSRAttr* new_attr, index_type block_size[2]) {
	index_type num_row = CSRAttrNumRow(csr);
	index_type num_col = CSRAttrNumCol(csr);
	index_type nnz = CSRAttrNNZ(csr);
	index_type i, j, k, l;
	index_type* row_ptr = CSRAttrRowPtr(csr);
	index_type* col_ind = CSRAttrColInd(csr);
	index_type* new_row_ptr = CSRAttrRowPtr(new_attr);
	index_type* new_col_ind = CSRAttrColInd(new_attr);
	index_type new_num_row = num_row * block_size[0];
	index_type new_num_col = num_col * block_size[1];
	index_type new_nnz = nnz * block_size[0] * block_size[1];
	new_row_ptr[0] = 0;
	for(i = 0; i < num_row; i++) {
		for(j = 0; j < block_size[0]; ++j) {
			new_row_ptr[i * block_size[0] + j + 1] = new_row_ptr[i * block_size[0] + j]
					+ (row_ptr[i + 1] - row_ptr[i]) * block_size[1];
		}
	}
	for(i = 0; i < num_row; i++) {
		for(k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
			for(l = 0; l < block_size[1]; l++) {
				new_col_ind[new_row_ptr[i * block_size[0]] + (k - row_ptr[i]) * block_size[1] + l] = col_ind[k] * block_size[1] + l;
			}
		}
		for(j = 1; j < block_size[0]; j++) {
			CdamMemcpy(new_col_ind + new_row_ptr[i * block_size[0] + j], new_col_ind + new_row_ptr[i * block_size[0]], (row_ptr[i + 1] - row_ptr[i]) * block_size[1] * sizeof(index_type), DEVICE_MEM, DEVICE_MEM);
		}
	}
}

static void ExpandCSRByBlockSize(const CSRAttr* csr, CSRAttr* new_attr, index_type block_size[2]) {
#ifdef CDAM_USE_CUDA
	ExpandCSRByBlockSizeDevice(csr, new_attr, block_size);
#else
	ExpandCSRByBlockSizeHost(csr, new_attr, block_size);
#endif
}

static void CSRAttrGetNZIndBatchedHost(const CSRAttr* attr, index_type batch_size,
																			 const index_type* row, const index_type* col,
																			 index_type* ind) {
}

#include "csr_impl.h"


CSRAttr* CSRAttrCreate(void* mesh) {
	CSRAttr* attr = CdamTMalloc(CSRAttr, 1, HOST_MEM);
	memset(attr, 0, sizeof(CSRAttr));
	index_type num_node = CdamMeshNumNode((CdamMesh*)mesh);
	index_type i, j;
	index_type nnz = 0;
	CSRAttrNumRow(attr) = num_node;
	CSRAttrNumCol(attr) = num_node;

	CSRHashMap* map = GetNodalGraphFromMesh((CdamMesh*)mesh);
	/* count nnz */
	for(i = 0; i < num_node; i++) {
		nnz += map->row_len[i];
	}
	CSRAttrNNZ(attr) = nnz;
	index_type* row_ptr = CdamTMalloc(index_type, num_node + 1, HOST_MEM);
	index_type* col_ind = CdamTMalloc(index_type, nnz, HOST_MEM);

	row_ptr[0] = 0;
	for (i = 0; i < num_node; i++) {
		row_ptr[i + 1] = row_ptr[i] + map->row_len[i];
		CdamMemcpy(col_ind + row_ptr[i], map->buff + i * PREALLOC_SIZE, sizeof(index_type) * map->row_len[i], HOST_MEM, HOST_MEM);
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

	index_type* row_ptr_dev = CdamTMalloc(index_type, num_node + 1, DEVICE_MEM);
	index_type* col_ind_dev = CdamTMalloc(index_type, nnz, DEVICE_MEM);
	CdamMemcpy(row_ptr_dev, row_ptr, sizeof(index_type) * (num_node + 1), DEVICE_MEM, HOST_MEM);
	CdamMemcpy(col_ind_dev, col_ind, sizeof(index_type) * nnz, DEVICE_MEM, HOST_MEM);
	CdamFree(row_ptr, sizeof(index_type) * (num_node + 1), HOST_MEM);
	CdamFree(col_ind, sizeof(index_type) * nnz, HOST_MEM);

	CSRAttrRowPtr(attr) = row_ptr_dev;
	CSRAttrColInd(attr) = col_ind_dev;
	return attr;
}


CSRAttr* CSRAttrCreateBlock(const CSRAttr* attr, index_type block_row, index_type block_col) {
	CSRAttr* new_attr = CdamTMalloc(CSRAttr, 1, HOST_MEM);
	memset(new_attr, 0, sizeof(CSRAttr));
	index_type nnz = CSRAttrNNZ(attr);
	index_type num_row = CSRAttrNumRow(attr);
	index_type num_col = CSRAttrNumCol(attr);

	CSRAttrNumRow(new_attr) = num_row * block_row;
	CSRAttrNumCol(new_attr) = num_col * block_col;

	CSRAttrNNZ(new_attr) = nnz * block_row * block_col;

	CSRAttrRowPtr(new_attr) = CdamTMalloc(index_type, num_row * block_row + 1, DEVICE_MEM);
	CSRAttrColInd(new_attr) = CdamTMalloc(index_type, nnz * block_row * block_col, DEVICE_MEM);
	new_attr->parent = attr;

	if(block_row == 1 && block_col == 1) {
		CdamMemcpy(CSRAttrRowPtr(new_attr), CSRAttrRowPtr(attr), sizeof(index_type) * (num_row + 1), DEVICE_MEM, DEVICE_MEM);
		CdamMemcpy(CSRAttrColInd(new_attr), CSRAttrColInd(attr), sizeof(index_type) * nnz, DEVICE_MEM, DEVICE_MEM);
	}
	else {
		index_type block_size[2] = {block_row, block_col};
		ExpandCSRByBlockSize(attr, new_attr, block_size);
	}
	return new_attr;
}

void GenerateSubmatCSRAttr(CSRAttr* attr, index_type nr, index_type* row,
													 index_type nc, index_type* col, CSRAttr** submat) {
#ifdef CDAM_USE_CUDA
	GenerateSubmatCSRAttrGPU(attr, nr, row, nc, col, submat);
#else
	*submat = CdamTMalloc(CSRAttr, 1, HOST_MEM);
	CdamMemset(*submat, 0, sizeof(CSRAttr), HOST_MEM);
	CSRAttrNumRow(*submat) = nr;
	CSRAttrNumCol(*submat) = nc;
	CSRAttrNNZ(*submat) = 0;

	index_type i, j, k;
	index_type nnz = 0;
	/* Count row length */
	CSRAttrRowPtr(*submat) = CdamTMalloc(index_type, nr + 1, DEVICE_MEM);
	CdamMemset(CSRAttrRowPtr(*submat), 0, sizeof(index_type) * (nr + 1), DEVICE_MEM);
	for(i = 0; i < nr; i++) {
		for(j = CSRAttrRowPtr(attr)[row[i]]; j < CSRAttrRowPtr(attr)[row[i] + 1]; j++) {
			for(k = 0; k < nc; k++) {
				if(CSRAttrColInd(attr)[j] == col[k]) {
					CSRAttrRowPtr(*submat)[i + 1]++;
					nnz++;
					break;
				}
			}
		}
	}
	/* Prefix sum */
	for(i = 0; i < nr; i++) {
		CSRAttrRowPtr(*submat)[i + 1] += CSRAttrRowPtr(*submat)[i];
	}
	/* Copy column indices */
	CSRAttrColInd(*submat) = CdamTMalloc(index_type, nnz, DEVICE_MEM);
	index_type* row_ptr = CSRAttrRowPtr(*submat);
	index_type* col_ind = CSRAttrColInd(*submat);

	for(i = 0; i < nr; i++) {
		index_type start = row_ptr[i];
		index_type end = row_ptr[i + 1];
		index_type count = 0;
		for(j = CSRAttrRowPtr(attr)[row[i]]; j < CSRAttrRowPtr(attr)[row[i] + 1]; j++) {
			for(k = 0; k < nc; k++) {
				if(CSRAttrColInd(attr)[j] == col[k]) {
					col_ind[start + count] = k;
					count++;
					break;
				}
			}
		}
	}

#endif
}

void CSRAttrDestroy(CSRAttr* attr) {
	CdamFree(CSRAttrRowPtr(attr), (CSRAttrNumRow(attr) + 1) * sizeof(index_type), DEVICE_MEM);
	CdamFree(CSRAttrColInd(attr), CSRAttrNNZ(attr) * sizeof(index_type), DEVICE_MEM);
	CdamFree(attr, sizeof(CSRAttr), HOST_MEM);
}

index_type CSRAttrLength(CSRAttr *attr, index_type row) {
	return CSRAttrRowPtr(attr)[row + 1] - CSRAttrRowPtr(attr)[row];
}

index_type* CSRAttrRow(CSRAttr *attr, index_type row) {
	return CSRAttrColInd(attr) + CSRAttrRowPtr(attr)[row];
}

void CSRAttrGetNonzeroIndBatched(const CSRAttr* attr, index_type batch_size,
																 const index_type* row, const index_type* col,
																 index_type* ind) {
#ifdef CDAM_USE_CUDA
	CSRAttrGetNZIndBatchedDevice(attr, batch_size, row, col, ind);
#else
	CSRAttrGetNZIndBatchedHost(attr, batch_size, row, col, ind);
#endif
}

__END_DECLS__

