
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "alloc.h"
#include "h5util.h"
#include "indexing.h"
#include "partition.h"
#include "Mesh.h"


__BEGIN_DECLS__

void CdamMeshCreate(MPI_Comm comm, CdamMesh** mesh) {
	*mesh = (CdamMesh*)CdamMallocHost(sizeof(CdamMesh));
	memset(*mesh, 0, sizeof(CdamMesh));
	(*mesh)->comm = comm;
	MPI_Comm_rank(comm, &(mesh[0]->rank));
}

void CdamMeshDestroy(CdamMesh* mesh) {
	index_type num_elem, ien_len;
	index_type num_facet = 0;
	num_elem = 0;
	num_elem += CdamMeshNumTet(mesh);
	num_elem += CdamMeshNumPrism(mesh);
	num_elem += CdamMeshNumHex(mesh);
	ien_len = 0;
	ien_len += CdamMeshNumTet(mesh) * 4;
	ien_len += CdamMeshNumPrism(mesh) * 6;
	ien_len += CdamMeshNumHex(mesh) * 8;
	// num_facet = mesh->bound_offset[CdamMeshNumBound(mesh)];
	CdamFree(CdamMeshCoord(mesh), sizeof(value_type) * CdamMeshNumNode(mesh) * 3, DEVICE_MEM);
	CdamFree(CdamMeshIEN(mesh), sizeof(index_type) * ien_len, DEVICE_MEM);

	// CdamFree(mesh->bound_id, sizeof(index_type) * CdamMeshNumBound(mesh), DEVICE_MEM);
	// CdamFree(mesh->bound_offset, sizeof(index_type) * (CdamMeshNumBound(mesh) + 1), DEVICE_MEM);
	// CdamFree(mesh->bound_f2e, sizeof(index_type) * tmp, DEVICE_MEM);
	// CdamFree(mesh->bound_forn, sizeof(index_type) * tmp, DEVICE_MEM);

	CdamFree(mesh->nodal_offset, sizeof(index_type) * (mesh->num_procs + 1), DEVICE_MEM);

	CdamFree(mesh->nodal_map_l2g_interior, sizeof(index_type) * CdamMeshNumNode(mesh), DEVICE_MEM);
	CdamFree(mesh->nodal_map_l2g_exterior, sizeof(index_type) * CdamMeshNumNode(mesh), DEVICE_MEM);

	CdamFree(mesh->elem_map_l2g, sizeof(index_type) * num_elem, DEVICE_MEM);

	CdamFree(mesh->color, sizeof(index_type) * num_elem, DEVICE_MEM);
	CdamFree(mesh->color_batch_offset, sizeof(index_type) * (mesh->num_color + 1), DEVICE_MEM);
	CdamFree(mesh->color_batch_ind, sizeof(index_type) * num_elem, DEVICE_MEM);
	// CdamFree(mesh->batch_offset, sizeof(index_type) * (mesh->num_batch + 1), DEVICE_MEM);
	// CdamFree(mesh->batch_ind, sizeof(index_type) * num_elem, DEVICE_MEM);
	CdamFree(mesh, sizeof(CdamMesh), HOST_MEM);
}

static void LoadCoord(H5FileInfo* h5f, const char* group_name, index_type num[], value_type** coord) {
	char dataset_name[256];
	ASSERT(strlen(group_name) < 192 && "Invalid group name: too long (>= 192).\n");

	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/xg", group_name);
	*coord = CdamTMalloc(value_type, num[0] * 3, HOST_MEM);
	H5ReadDatasetVal(h5f, dataset_name, *coord);
}

static void LoadElementConnectivity(H5FileInfo* h5f, const char* group_name, index_type num[], index_type** ien) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	char dataset_name[256];
	ASSERT(strlen(group_name) < 192 && "Invalid group name: too long (>= 192).\n");

	/* Read the number of nodes */
	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/xg", group_name);
	H5GetDatasetSize(h5f, dataset_name, num);
	num[0] = num[0] / 3;

	/* Read the number of elements */
	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/ien/tet", group_name);
	H5GetDatasetSize(h5f, dataset_name, num + 1);

	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/ien/prism", group_name);
	H5GetDatasetSize(h5f, dataset_name, num + 2);

	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/ien/hex", group_name);
	H5GetDatasetSize(h5f, dataset_name, num + 3);

	/* Allocate memory for element connectivity */
	index_type* eind = (index_type*)CdamMallocHost(sizeof(index_type) * (num[1] + num[2] + num[3]));
	num[1] /= 4; 
	num[2] /= 6;
	num[3] /= 8;

	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/ien/tet", group_name);
	H5ReadDatasetInd(h5f, dataset_name, eind);

	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/ien/prism", group_name);
	H5ReadDatasetInd(h5f, dataset_name, eind + num[1] * 4);

	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/ien/hex", group_name);
	H5ReadDatasetInd(h5f, dataset_name, eind + num[1] * 4 + num[2] * 6);

	*ien = eind;
}

/*
struct cdam_mesh_part_struct {
	index_type num_node_owned;
	index_type num_node_ghosted;

	index_type* node_map_l2g;

	index_type num_elem_owned;
	index_type* elem_map_l2g;
};
typedef struct cdam_mesh_part_struct CdamMeshPart;
*/

struct QsortCtx {
	index_type* ien;
	index_type* epart;
	index_type size;
};


static void* qsort_ctx = NULL;
static int QsortCmpElem(const void* a, const void* b) {
	const index_type* ien = ((struct QsortCtx*)qsort_ctx)->ien;
	const index_type* epart = ((struct QsortCtx*)qsort_ctx)->epart;
	index_type size = ((struct QsortCtx*)qsort_ctx)->size;

	const index_type* ia = (index_type*)a;
	const index_type* ib = (index_type*)b;

	return epart[(ia - ien) / size] - epart[(ib - ien) / size];
}

static void ShuffleIENByPartition(index_type num[], index_type* ien, index_type* epart) {
	index_type i, j;

	qsort_ctx = &(struct QsortCtx){ien, epart, 4};
	qsort(ien, num[1], 4, QsortCmpElem);

	qsort_ctx = &(struct QsortCtx){ien + num[1] * 4, epart + num[1], 6};
	qsort(ien + num[1] * 4, num[2], 6, QsortCmpElem);

	qsort_ctx = &(struct QsortCtx){ien + num[1] * 4 + num[2] * 6, epart + num[1] + num[2], 8};
	qsort(ien + num[1] * 4 + num[2] * 6, num[3], 8, QsortCmpElem);

	qsort_ctx = NULL;
} 


static int QsortCmpIndexType(const void* a, const void* b) {
	return *((const index_type*)a) - *((const index_type*)b);
}
static index_type CountDistinctEntry(index_type* array, index_type size, index_type* array_set) {
	index_type i, j;
	index_type count;
	index_type* array_sorted = CdamTMalloc(index_type, size, HOST_MEM);
	CdamMemcpy(array_sorted, array, sizeof(index_type) * size, HOST_MEM, HOST_MEM);
	qsort(array_sorted, size, sizeof(index_type), QsortCmpIndexType);

	count = 0;
	for(i = 0; i < size; ++i) {
		if(i == 0 || array_sorted[i] != array_sorted[i - 1]) {
			count++;
		}
	}
	if(array_set) {
		for(i = 0, j = 0; i < size; ++i) {
			if(i == 0 || array_sorted[i] != array_sorted[i - 1]) {
				array_set[j++] = array_sorted[i];
			}
		}
	}
	return count;
}

static int QsortCmpNpart(const void* a, const void* b) {
	index_type* npart = (index_type*)((struct QsortCtx*)qsort_ctx)->epart;
	index_type rank = ((struct QsortCtx*)qsort_ctx)->size;
	index_type ia = *(const index_type*)a;
	index_type ib = *(const index_type*)b;
	b32 is_owned_a = npart[ia] == rank;
	b32 is_owned_b = npart[ib] == rank;

	if(is_owned_b - is_owned_a) {
		return is_owned_b - is_owned_a;
	}
	else {
		return npart[ia] - npart[ib];
	}
}

void CdamMeshLoad(CdamMesh* mesh, H5FileInfo* h5f, const char* group_name) {
	index_type i;
	int rank, size;
	char dataset_name[256];
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	index_type* epart_count = NULL;
	index_type* count_npart = NULL;
	index_type num[4];
	index_type* ien = NULL;
	index_type* epart = NULL;
	index_type* npart = NULL;

	value_type* coord = NULL;

	epart_count = CdamTMalloc(index_type, (size + 1) * 3, HOST_MEM);
	CdamMemset(epart_count, 0, sizeof(index_type) * (size + 1) * 3, HOST_MEM);
	if(rank == 0) {
		/* 0. Load element connectivity */
		LoadElementConnectivity(h5f, group_name, num, &ien);
		epart = CdamTMalloc(index_type, num[1] + num[2] + num[3], HOST_MEM);
		npart = CdamTMalloc(index_type, num[0], HOST_MEM);



		/* 1. Metis partition */
		PartitionMeshMetis(num, ien, size, epart, npart);

		/* 2. Shuffle the element connectivity */
		ShuffleIENByPartition(num, ien, epart);
		/* 2. Prepare the data for other processes */

		for(i = 0; i < num[1]; ++i) {
			epart_count[size * 0 + epart[i] + 1]++;
		}
		for(i = 0; i < num[2]; ++i) {
			epart_count[size * 1 + epart[num[1] + i] + 1]++;
		}
		for(i = 0; i < num[3]; ++i) {
			epart_count[size * 2 + epart[num[1] + num[2] + i] + 1]++;
		}
		for(i = 0; i < size; ++i) {
			epart_count[size * 0 + i + 1] += epart_count[size * 0 + i];
		}
		for(i = 0; i < size; ++i) {
			epart_count[size * 1 + i + 1] += epart_count[size * 1 + i];
		}
		for(i = 0; i < size; ++i) {
			epart_count[size * 2 + i + 1] += epart_count[size * 2 + i];
		}
	}

	MPI_Bcast(num, sizeof(index_type) * 4, MPI_CHAR, 0, mesh->comm);
	if(rank) {
		npart = CdamTMalloc(index_type, num[0], HOST_MEM);
	}
	MPI_Bcast(npart, sizeof(index_type) * num[0], MPI_CHAR, 0, mesh->comm);

	MPI_Bcast(epart_count, sizeof(index_type) * (size + 1) * 3, MPI_CHAR, 0, mesh->comm);

	mesh->num[0] = num[0];
	mesh->num[1] = epart_count[size * 0 + rank + 1] - epart_count[size * 0 + rank];
	mesh->num[2] = epart_count[size * 1 + rank + 1] - epart_count[size * 1 + rank];
	mesh->num[3] = epart_count[size * 2 + rank + 1] - epart_count[size * 2 + rank];

	CdamMeshIEN(mesh) = CdamTMalloc(index_type, mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8, HOST_MEM);
	
	int* send_count = CdamTMalloc(int, size, HOST_MEM);
	int* send_disp = CdamTMalloc(int, size, HOST_MEM);

	for(i = 0; i < size; ++i) {
		send_count[i] = (epart_count[size * 0 + i + 1] - epart_count[size * 0 + i]) * 4 * sizeof(index_type);
		send_disp[i] = epart_count[size * 0 + i] * 4 * sizeof(index_type);
	}
	MPI_Scatterv(ien, send_count, send_disp, MPI_CHAR,
							 mesh->ien, mesh->num[1] * 4 * sizeof(index_type), MPI_CHAR,
							 0, mesh->comm); 
	for(i = 0; i < size; ++i) {
		send_count[i] = (epart_count[size * 1 + i + 1] - epart_count[size * 1 + i]) * 6 * sizeof(index_type);
		send_disp[i] = epart_count[size * 1 + i] * 6 * sizeof(index_type);
	}
	MPI_Scatterv(ien + num[1] * 4, send_count, send_disp, MPI_CHAR,
							 mesh->ien + mesh->num[1] * 4, mesh->num[2] * 6 * sizeof(index_type), MPI_CHAR,
							 0, mesh->comm);
	for(i = 0; i < size; ++i) {
		send_count[i] = (epart_count[size * 2 + i + 1] - epart_count[size * 2 + i]) * 8 * sizeof(index_type);
		send_disp[i] = epart_count[size * 2 + i] * 8 * sizeof(index_type);
	}
	MPI_Scatterv(ien + num[1] * 4 + num[2] * 6, send_count, send_disp, MPI_CHAR,
							 mesh->ien + mesh->num[1] * 4 + mesh->num[2] * 6, mesh->num[3] * 8 * sizeof(index_type), MPI_CHAR,
							 0, mesh->comm);


	mesh->rank = rank;
	mesh->num_procs = size;

	mesh->nodal_offset = CdamTMalloc(index_type, size + 1, HOST_MEM);
	CdamMemset(mesh->nodal_offset, 0, sizeof(index_type) * (size + 1), HOST_MEM);

	for(i = 0; i < num[0]; ++i) {
		mesh->nodal_offset[npart[i] + 1]++;
	}
	for(i = 0; i < size; ++i) {
		mesh->nodal_offset[i + 1] += mesh->nodal_offset[i];
	}

	mesh->num[0] = CountDistinctEntry(mesh->ien, mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8, NULL);
	mesh->nodal_map_l2g_interior = (index_type*)CdamMallocHost(sizeof(index_type) * mesh->num[0]);
	mesh->nodal_map_l2g_exterior = (index_type*)CdamMallocHost(sizeof(index_type) * mesh->num[0]);
	CountDistinctEntry(mesh->ien, mesh->num[0] * 4 + mesh->num[1] * 6 + mesh->num[2] * 8, mesh->nodal_map_l2g_exterior);

	qsort_ctx = &(struct QsortCtx){npart, NULL, size};
	qsort(mesh->nodal_map_l2g_exterior, mesh->num[0], sizeof(index_type), QsortCmpNpart);


	/* Generate the l2g_interior */
	int* send_node_count = CdamTMalloc(int, size * 2 + 1, HOST_MEM);
	int* send_node_offset = send_node_count + size;
	int* recv_node_count = CdamTMalloc(int, size * 2 + 1, HOST_MEM);
	int* recv_node_offset = recv_node_count + size;

	CdamMemset(send_node_count, 0, sizeof(int) * (size * 2 + 1), HOST_MEM);
	CdamMemset(recv_node_count, 0, sizeof(int) * (size * 2 + 1), HOST_MEM);
	index_type num_owned_node = mesh->nodal_offset[rank + 1] - mesh->nodal_offset[rank];
	for(i = 0; i < num[0] - num_owned_node; ++i) {
		send_node_count[npart[mesh->nodal_map_l2g_exterior[i + num_owned_node]]] += sizeof(int);
	}
	MPI_Alltoall(send_node_count, sizeof(int), MPI_CHAR,
							 recv_node_count, sizeof(int), MPI_CHAR, mesh->comm);
	for(i = 0; i < size; ++i) {
		send_node_offset[i + 1] = send_node_count[i] + send_node_offset[i];
		recv_node_offset[i + 1] = recv_node_count[i] + recv_node_offset[i];
	}
	index_type* node_index_buff = CdamTMalloc(index_type, recv_node_offset[size], HOST_MEM);
	MPI_Alltoallv(mesh->nodal_map_l2g_exterior + num_owned_node, send_node_count, send_node_offset, MPI_CHAR,
							  node_index_buff, recv_node_count, recv_node_offset, MPI_CHAR, mesh->comm);

	for(i = 0; i < recv_node_offset[size]; ++i) {
		/* find the local index in mesh->nodal_map_l2g_exterior */
		index_type* p = (index_type*)bsearch(node_index_buff + i, mesh->nodal_map_l2g_exterior,
																				 num[0] - num_owned_node, sizeof(index_type),
																				 QsortCmpIndexType);
		node_index_buff[i] = p - mesh->nodal_map_l2g_exterior;
	}
	MPI_Alltoallv(node_index_buff, recv_node_count,
								recv_node_offset, MPI_CHAR,
							  mesh->nodal_map_l2g_interior + num_owned_node,	send_node_count,
								send_node_offset, MPI_CHAR, mesh->comm);

	for(i = 0; i < num_owned_node; ++i) {
		mesh->nodal_map_l2g_interior[i] = i + mesh->nodal_offset[rank];
	}
	for(i = 0; i < num[0] - num_owned_node; ++i) {
		index_type part = npart[mesh->nodal_map_l2g_interior[i + num_owned_node]];
		mesh->nodal_map_l2g_interior[i + num_owned_node] += mesh->nodal_offset[part];
	}

	index_type *block_len = CdamTMalloc(index_type, num[0], HOST_MEM);
	for(i = 0; i < num[0]; ++i) {
		block_len[i] = 3;
	}

	CdamMeshCoord(mesh) = CdamTMalloc(value_type, num[0] * 3, HOST_MEM);
	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/xg", group_name);
	H5ReadDatasetValIndexed(h5f, group_name, num[0],
													block_len, mesh->nodal_map_l2g_interior, CdamMeshCoord(mesh));

	CdamFree(block_len, sizeof(index_type) * num[0], HOST_MEM);
	CdamFree(epart_count, sizeof(index_type) * (size + 1) * 3, HOST_MEM);
	CdamFree(epart, sizeof(index_type) * (num[1] + num[2] + num[3]), HOST_MEM);
	CdamFree(npart, sizeof(index_type) * num[0], HOST_MEM);
	CdamFree(ien, sizeof(index_type) * (num[1] + num[2] + num[3]), HOST_MEM);
	CdamFree(send_count, sizeof(int) * size, HOST_MEM);
	CdamFree(send_disp, sizeof(int) * size, HOST_MEM);
	CdamFree(send_node_count, sizeof(int) * (size * 2 + 1), HOST_MEM);
	CdamFree(recv_node_count, sizeof(int) * (size * 2 + 1), HOST_MEM);
	CdamFree(node_index_buff, sizeof(index_type) * recv_node_offset[size], HOST_MEM);

}

static void MoveToDevice(void** ptr, size_t size) {
	if(*ptr == NULL) {
		return;
	}
	void* ptr_dev = CdamTMalloc(byte, size, DEVICE_MEM);
	CdamMemcpy(ptr_dev, *ptr, size, HOST_MEM, DEVICE_MEM);
	CdamFree(*ptr, size, HOST_MEM);
	*ptr = ptr_dev;
}

void CdamMeshToDevice(CdamMesh *mesh) {
	index_type num_node = mesh->num[0];
	index_type num_tet = mesh->num[1];
	index_type num_prism = mesh->num[2];
	index_type num_hex = mesh->num[3];

	MoveToDevice((void**)&(mesh->coord), sizeof(value_type) * num_node * 3);
	MoveToDevice((void**)&(mesh->ien), sizeof(index_type) * (num_tet * 4 + num_prism * 6 + num_hex * 8));

	MoveToDevice((void**)&(mesh->nodal_offset), sizeof(index_type) * (mesh->num_procs + 1));
	MoveToDevice((void**)&(mesh->nodal_map_l2g_interior), sizeof(index_type) * num_node);
	MoveToDevice((void**)&(mesh->nodal_map_l2g_exterior), sizeof(index_type) * num_node);

	MoveToDevice((void**)&(mesh->color), sizeof(index_type) * (num_tet + num_prism + num_hex));
	MoveToDevice((void**)&(mesh->color_batch_offset), sizeof(index_type) * (mesh->num_color + 1));
	MoveToDevice((void**)&(mesh->color_batch_ind), sizeof(index_type) * (num_tet + num_prism + num_hex));

}

void CdamMeshColor(CdamMesh* mesh, Arena scratch) {
	index_type i;
	/* Assign color to each element */
	index_type num_elem = mesh->num[1] + mesh->num[2] + mesh->num[3];
	mesh->color = CdamTMalloc(index_type, num_elem, DEVICE_MEM);
	ColorMeshTet(mesh, MAX_COLOR, mesh->color, scratch);

	/* Sort the element id by color */
	index_type max_color = GetMaxColor(mesh->color, num_elem);
	mesh->color_batch_offset = CdamTMalloc(index_type, max_color + 1, HOST_MEM);
	mesh->color_batch_offset[0] = 0;
	mesh->color_batch_ind = CdamTMalloc(index_type, num_elem, HOST_MEM);

	for(i = 0; i < max_color; ++i) {
		index_type count = CountValueI(mesh->color, num_elem, i, scratch);
		mesh->color_batch_offset[i + 1] = mesh->color_batch_offset[i] + count;
		// FilterValueI(mesh->color, num_elem, i, mesh->color_batch_ind + mesh->color_batch_offset[i]);	
	}
	/* iota to mesh->color_batch_ind */
	for(i = 0; i < num_elem; ++i) {
		mesh->color_batch_ind[i] = i;
	}
	MoveToDevice((void**)&(mesh->color_batch_ind), sizeof(index_type) * num_elem);
	/* Sort the element id by color */
	SortByKeyI(mesh->color, mesh->color_batch_ind, num_elem, scratch);
}


__END_DECLS__

