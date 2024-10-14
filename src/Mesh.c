
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
	*mesh = CdamTMalloc(CdamMesh, 1, HOST_MEM);
	memset(*mesh, 0, sizeof(CdamMesh));
	(*mesh)->comm = comm;
	MPI_Comm_rank(comm, &(mesh[0]->rank));
	MPI_Comm_size(comm, &(mesh[0]->num_procs));
}

void CdamMeshDestroy(CdamMesh* mesh) {
	index_type num_elem, ien_len;
	// index_type num_facet = 0;
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

	CdamFree(mesh->color, sizeof(index_type) * num_elem, DEVICE_MEM);
	CdamFree(mesh->color_batch_offset, sizeof(index_type) * (mesh->num_color + 1), HOST_MEM);
	CdamFree(mesh->color_batch_ind, sizeof(index_type) * num_elem, DEVICE_MEM);
	// CdamFree(mesh->batch_offset, sizeof(index_type) * (mesh->num_batch + 1), DEVICE_MEM);
	// CdamFree(mesh->batch_ind, sizeof(index_type) * num_elem, DEVICE_MEM);
	CdamFree(mesh, sizeof(CdamMesh), HOST_MEM);
}

static void LoadCoord(H5FileInfo* h5f, const char* group_name, index_type num_node, index_type*index, value_type* coord) {
	char dataset_name[256];
	ASSERT(strlen(group_name) < 192 && "Invalid group name: too long (>= 192).\n");

	snprintf(dataset_name, sizeof(dataset_name), "%s/xg", group_name);
	index_type* hindex = CdamTMalloc(index_type, num_node * 3, HOST_MEM);
	index_type i;
	for(i = 0; i < num_node; i++) {
		hindex[i * 3 + 0] = index[i] * 3 + 0;
		hindex[i * 3 + 1] = index[i] * 3 + 1;
		hindex[i * 3 + 2] = index[i] * 3 + 2;
	}

	H5ReadDatasetValIndexed(h5f, dataset_name, num_node * 3, hindex, coord);
	CdamFree(index, sizeof(index_type) * num_node, HOST_MEM);
}

static void LoadElementConnectivity(H5FileInfo* h5f, const char* group_name, index_type num[], index_type** ien) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	char dataset_name[256];
	ASSERT(strlen(group_name) < 192 && "Invalid group name: too long (>= 192).\n");

	/* Read the number of nodes */
	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/xg", group_name);
	if(H5DatasetExist(h5f, dataset_name) == 0) {
		fprintf(stderr, "Dataset %s does not exist.\n", dataset_name);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	H5GetDatasetSize(h5f, dataset_name, num);
	num[0] = num[0] / 3;

	/* Read the number of elements */
	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/ien/tet", group_name);
	if(H5DatasetExist(h5f, dataset_name) == 0) {
		fprintf(stderr, "Dataset %s does not exist.\n", dataset_name);
		num[1] = 0;
	}
	else {
		H5GetDatasetSize(h5f, dataset_name, num + 1);
	}

	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/ien/prism", group_name);
	if(H5DatasetExist(h5f, dataset_name) == 0) {
		fprintf(stderr, "Dataset %s does not exist.\n", dataset_name);
		num[2] = 0;
	}
	else {
		H5GetDatasetSize(h5f, dataset_name, num + 2);
	}

	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/ien/hex", group_name);
	if(H5DatasetExist(h5f, dataset_name) == 0) {
		fprintf(stderr, "Dataset %s does not exist.\n", dataset_name);
		num[3] = 0;
	}
	else {
		H5GetDatasetSize(h5f, dataset_name, num + 3);
	}

	/* Allocate memory for element connectivity */
	index_type* eind = CdamTMalloc(index_type, num[1] + num[2] + num[3], HOST_MEM);
	num[1] /= 4; 
	num[2] /= 6;
	num[3] /= 8;

	if(num[1] > 0) {
		snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/ien/tet", group_name);
		H5ReadDatasetInd(h5f, dataset_name, eind);
	}

	if(num[2] > 0) {
		snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/ien/prism", group_name);
		H5ReadDatasetInd(h5f, dataset_name, eind + num[1] * 4);
	}

	if(num[3] > 0) {
		snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/ien/hex", group_name);
		H5ReadDatasetInd(h5f, dataset_name, eind + num[1] * 4 + num[2] * 6);
	}

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
	index_type* begin;
	index_type* key;
	index_type stride;
};


static void* qsort_ctx = NULL;
static int QsortCmpElem(const void* a, const void* b) {
	const index_type* key = ((struct QsortCtx*)qsort_ctx)->key;

	index_type index_a = *(const index_type*)a;
	index_type index_b = *(const index_type*)b;

	index_type key_val_a = key[index_a];
	index_type key_val_b = key[index_b];

	if(key_val_a != key_val_b) {
		return (key_val_a > key_val_b) - (key_val_a < key_val_b);
	}
	else {
		return (index_a > index_b) - (index_a < index_b);
	}
}

static void ShuffleIENByPartition(index_type num[], index_type* ien, index_type* epart) {
	index_type i;

	struct QsortCtx ctx = {NULL, NULL, 0};
	qsort_ctx = &ctx;

	index_type buffer_size = MAX(num[1] * 5, MAX(num[2] * 7, num[3] * 9));
	index_type* ien_copy = CdamTMalloc(index_type, buffer_size, HOST_MEM);

	for(i = 0; i < num[1]; i++) {
		ien_copy[i * 5 + 0] = i;
		ien_copy[i * 5 + 1] = ien[i * 4 + 0];
		ien_copy[i * 5 + 2] = ien[i * 4 + 1];
		ien_copy[i * 5 + 3] = ien[i * 4 + 2];
		ien_copy[i * 5 + 4] = ien[i * 4 + 3];
	}
	ctx.key = epart;
	qsort(ien_copy, num[1], 5 * sizeof(index_type), QsortCmpElem);
	for(i = 0; i < num[1]; ++i) {
		ien[i * 4 + 0] = ien_copy[i * 5 + 1];
		ien[i * 4 + 1] = ien_copy[i * 5 + 2];
		ien[i * 4 + 2] = ien_copy[i * 5 + 3];
		ien[i * 4 + 3] = ien_copy[i * 5 + 4];
	}

	for(i = 0; i < num[2]; ++i) {
		ien_copy[i * 7 + 0] = i;
		ien_copy[i * 7 + 1] = ien[num[1] * 4 + i * 6 + 0];
		ien_copy[i * 7 + 2] = ien[num[1] * 4 + i * 6 + 1];
		ien_copy[i * 7 + 3] = ien[num[1] * 4 + i * 6 + 2];
		ien_copy[i * 7 + 4] = ien[num[1] * 4 + i * 6 + 3];
		ien_copy[i * 7 + 5] = ien[num[1] * 4 + i * 6 + 4];
		ien_copy[i * 7 + 6] = ien[num[1] * 4 + i * 6 + 5];
	}
	ctx.key = epart + num[1];
	qsort(ien_copy, num[2], 7 * sizeof(index_type), QsortCmpElem);
	for(i = 0; i < num[2]; ++i) {
		ien[num[1] * 4 + i * 6 + 0] = ien_copy[i * 7 + 1];
		ien[num[1] * 4 + i * 6 + 1] = ien_copy[i * 7 + 2];
		ien[num[1] * 4 + i * 6 + 2] = ien_copy[i * 7 + 3];
		ien[num[1] * 4 + i * 6 + 3] = ien_copy[i * 7 + 4];
		ien[num[1] * 4 + i * 6 + 4] = ien_copy[i * 7 + 5];
		ien[num[1] * 4 + i * 6 + 5] = ien_copy[i * 7 + 6];
	}

	for(i = 0; i < num[3]; ++i) {
		ien_copy[i * 9 + 0] = i;
		ien_copy[i * 9 + 1] = ien[num[1] * 4 + num[2] * 6 + i * 8 + 0];
		ien_copy[i * 9 + 2] = ien[num[1] * 4 + num[2] * 6 + i * 8 + 1];
		ien_copy[i * 9 + 3] = ien[num[1] * 4 + num[2] * 6 + i * 8 + 2];
		ien_copy[i * 9 + 4] = ien[num[1] * 4 + num[2] * 6 + i * 8 + 3];
		ien_copy[i * 9 + 5] = ien[num[1] * 4 + num[2] * 6 + i * 8 + 4];
		ien_copy[i * 9 + 6] = ien[num[1] * 4 + num[2] * 6 + i * 8 + 5];
		ien_copy[i * 9 + 7] = ien[num[1] * 4 + num[2] * 6 + i * 8 + 6];
		ien_copy[i * 9 + 8] = ien[num[1] * 4 + num[2] * 6 + i * 8 + 7];
	}
	ctx.key = epart + num[1] + num[2];
	qsort(ien + num[1] * 4 + num[2] * 6, num[3], 8 * sizeof(index_type), QsortCmpElem);
	for(i = 0; i < num[3]; ++i) {
		ien[num[1] * 4 + num[2] * 6 + i * 8 + 0] = ien_copy[i * 9 + 1];
		ien[num[1] * 4 + num[2] * 6 + i * 8 + 1] = ien_copy[i * 9 + 2];
		ien[num[1] * 4 + num[2] * 6 + i * 8 + 2] = ien_copy[i * 9 + 3];
		ien[num[1] * 4 + num[2] * 6 + i * 8 + 3] = ien_copy[i * 9 + 4];
		ien[num[1] * 4 + num[2] * 6 + i * 8 + 4] = ien_copy[i * 9 + 5];
		ien[num[1] * 4 + num[2] * 6 + i * 8 + 5] = ien_copy[i * 9 + 6];
		ien[num[1] * 4 + num[2] * 6 + i * 8 + 6] = ien_copy[i * 9 + 7];
	}

	CdamFree(ien_copy, sizeof(index_type) * buffer_size, HOST_MEM);
	qsort_ctx = NULL;
} 


static int QsortCmpIndexType(const void* a, const void* b) {
	index_type ia = *(const index_type*)a;
	index_type ib = *(const index_type*)b;

	return (ia > ib) - (ia < ib);
}


static index_type CountDistinctEntry(index_type* array, index_type size, index_type* array_set) {
	index_type i, j;
	index_type count;
	index_type* array_sorted = CdamTMalloc(index_type, size, HOST_MEM);
	CdamMemcpy(array_sorted, array, sizeof(index_type) * size, HOST_MEM, HOST_MEM);
	qsort(array_sorted, size, sizeof(index_type), QsortCmpIndexType);

	/*
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank == 0) {
		FILE* fp = fopen("array_sorted.txt", "w");
		for(i = 0; i < size; ++i) {
			fprintf(fp, "%d\n", array_sorted[i]);
		}
		fclose(fp);
	}
	*/

	count = 0;
	for(i = 0; i < size; ++i) {
		if(i == 0 || array_sorted[i] != array_sorted[i - 1]) {
			count++;
		}
	}
	if(array_set) {
		j = 0;
		for(i = 0; i < size; ++i) {
			if(i == 0 || array_sorted[i] != array_sorted[i - 1]) {
				array_set[j++] = array_sorted[i];
			}
		}
	}
	CdamFree(array_sorted, sizeof(index_type) * size, HOST_MEM);
	return count;
}

static int QsortCmpNpart(const void* a, const void* b) {
	const index_type* is_shared = ((struct QsortCtx*)qsort_ctx)->begin;
	const index_type* npart = ((struct QsortCtx*)qsort_ctx)->key;
	int rank = ((struct QsortCtx*)qsort_ctx)->stride;
	index_type ia = *(const index_type*)a;
	index_type ib = *(const index_type*)b;
	
	b32 is_owned_a = npart[ia] == rank ? 0 : 1;
	b32 is_owned_b = npart[ib] == rank ? 0 : 1;

	if(is_owned_a != is_owned_b) {
		return is_owned_a - is_owned_b;
	}
	else if(is_owned_a) {
		return is_shared[ia] - is_shared[ib];
	}
	else {
		return (npart[ia] > npart[ib]) - (npart[ia] < npart[ib]);
	}
	
}

struct SerialMesh {
	index_type num[4];
	index_type* ien;
	index_type* epart;
	index_type* npart;
};

typedef struct SerialMesh SerialMesh;

static void ReadSerialMesh(H5FileInfo* h5f, const char* group_name, SerialMesh* mesh) {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank) {
		return;
	}

	char  dataset_name[256];
	ASSERT(strlen(group_name) < 192 && "Invalid group name: too long (>= 192).\n");

	/* Read the number of nodes */
	snprintf(dataset_name, sizeof(dataset_name), "%s/xg", group_name);
	if(H5DatasetExist(h5f, dataset_name) == 0) {
		fprintf(stderr, "Dataset %s does not exist.\n", dataset_name);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	H5GetDatasetSize(h5f, dataset_name, mesh->num);
	mesh->num[0] /= 3;

	/* Read the number of elements */
	snprintf(dataset_name, sizeof(dataset_name), "%s/ien/tet", group_name);
	if(H5DatasetExist(h5f, dataset_name) == 0) {
		fprintf(stderr, "Dataset %s does not exist.\n", dataset_name);
		mesh->num[1] = 0;
	}
	else {
		H5GetDatasetSize(h5f, dataset_name, mesh->num + 1);
		mesh->num[1] /= 4;
	}

	snprintf(dataset_name, sizeof(dataset_name), "%s/ien/prism", group_name);
	if(H5DatasetExist(h5f, dataset_name) == 0) {
		fprintf(stderr, "Dataset %s does not exist.\n", dataset_name);
		mesh->num[2] = 0;
	}
	else {
		H5GetDatasetSize(h5f, dataset_name, mesh->num + 2);
		mesh->num[2] /= 6;
	}

	snprintf(dataset_name, sizeof(dataset_name), "%s/ien/hex", group_name);
	if(H5DatasetExist(h5f, dataset_name) == 0) {
		fprintf(stderr, "Dataset %s does not exist.\n", dataset_name);
		mesh->num[3] = 0;
	}
	else {
		H5GetDatasetSize(h5f, dataset_name, mesh->num + 3);
		mesh->num[3] /= 8;
	}

	/* Allocate memory for element connectivity */
	mesh->ien = CdamTMalloc(index_type, mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8, HOST_MEM);

	if(mesh->num[1] > 0) {
		snprintf(dataset_name, sizeof(dataset_name), "%s/ien/tet", group_name);
		H5ReadDatasetInd(h5f, dataset_name, mesh->ien);
	}

	if(mesh->num[2] > 0) {
		snprintf(dataset_name, sizeof(dataset_name), "%s/ien/prism", group_name);
		H5ReadDatasetInd(h5f, dataset_name, mesh->ien + mesh->num[1] * 4);
	}

	if(mesh->num[3] > 0) {
		snprintf(dataset_name, sizeof(dataset_name), "%s/ien/hex", group_name);
		H5ReadDatasetInd(h5f, dataset_name, mesh->ien + mesh->num[1] * 4 + mesh->num[2] * 6);
	}

	/* Allocate memory for partition */
	mesh->epart = CdamTMalloc(index_type, mesh->num[1] + mesh->num[2] + mesh->num[3], HOST_MEM);
	mesh->npart = CdamTMalloc(index_type, mesh->num[0], HOST_MEM);
}

static void PartitionMesh(SerialMesh* smesh, index_type* epart, index_type* npart) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if(rank == 0) {
		PartitionMeshMetis(smesh->num, smesh->ien, size, smesh->epart, smesh->npart);
	}


}

static void ShuffleElemByPartition(index_type nelem, index_type*ien, index_type nshl, index_type* epart) {
	struct QsortCtx ctx = {NULL, epart, 0};
	qsort_ctx = &ctx;

	index_type i, j;
	index_type buffer_size = nelem * (nshl + 1);
	index_type* ien_copy = CdamTMalloc(index_type, buffer_size, HOST_MEM);

	for(i = 0; i < nelem; ++i) {
		ien_copy[i * (nshl + 1) + 0] = i;
		for(j = 0; j < nshl; ++j) {
			ien_copy[i * (nshl + 1) + j + 1] = ien[i * nshl + j];
		}
	}

	qsort(ien_copy, nelem, (nshl + 1) * sizeof(index_type), QsortCmpElem);

	for(i = 0; i < nelem; ++i) {
		for(j = 0; j < nshl; ++j) {
			ien[i * nshl + j] = ien_copy[i * (nshl + 1) + j + 1];
		}
	}

}

static index_type DistributeElem(index_type nelem, index_type* ien, index_type* epart, index_type* mesh_ien, index_type nshl) {
	int rank, size;
	index_type i;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	index_type *send_count = CdamTMalloc(index_type, size, HOST_MEM);
	index_type *send_displ = CdamTMalloc(index_type, size, HOST_MEM);
	if(rank == 0) {
		ShuffleElemByPartition(nelem, ien, nshl, epart);
		CdamMemset(send_count, 0, sizeof(index_type) * size, HOST_MEM);
		for(i = 0; i < nelem; ++i) {
			send_count[epart[i]]++;
		}
		for(i = 0; i < size; ++i) {
			send_count[i] *= nshl * sizeof(index_type);
		}
		send_displ[0] = 0;
		for(i = 0; i < size - 1; i++) {
			send_displ[i + 1] = send_displ[i] + send_count[i];
		}
	}

	MPI_Bcast(send_count, sizeof(index_type) * size, MPI_CHAR, 0, MPI_COMM_WORLD);
	if(mesh_ien) {
		MPI_Scatterv(ien, send_count, send_displ, MPI_CHAR,
								 mesh_ien, send_count[rank], MPI_CHAR,
								 0, MPI_COMM_WORLD);
	}
	CdamFree(send_count, sizeof(index_type) * size, HOST_MEM);
	CdamFree(send_displ, sizeof(index_type) * size, HOST_MEM);
	return send_count[rank] / (nshl * sizeof(index_type));
}

static void DistributeMesh(CdamMesh* mesh, SerialMesh* smesh) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	mesh->num[1] = DistributeElem(smesh->num[1], smesh->ien, smesh->epart, NULL, 4);
	mesh->num[2] = DistributeElem(smesh->num[2], smesh->ien + smesh->num[1] * 4, smesh->epart + smesh->num[1], NULL, 6);
	mesh->num[3] = DistributeElem(smesh->num[3], smesh->ien + smesh->num[1] * 4 + smesh->num[2] * 6, smesh->epart + smesh->num[1] + smesh->num[2], NULL, 8);

	CdamMeshIEN(mesh) = CdamTMalloc(index_type, mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8, HOST_MEM);

	DistributeElem(smesh->num[1], smesh->ien, smesh->epart, CdamMeshIEN(mesh), 4);
	DistributeElem(smesh->num[2], smesh->ien + smesh->num[1] * 4, smesh->epart + smesh->num[1], CdamMeshIEN(mesh) + mesh->num[1] * 4, 6);
	DistributeElem(smesh->num[3], smesh->ien + smesh->num[1] * 4 + smesh->num[2] * 6, smesh->epart + smesh->num[1] + smesh->num[2], CdamMeshIEN(mesh) + mesh->num[1] * 4 + mesh->num[2] * 6, 8);

}

static void GenerateL2GMapExterior(CdamMesh* mesh, SerialMesh* smesh) {
	index_type i;
	/* Count the number of distinct nodes */
	mesh->num[0] = CountDistinctEntry(CdamMeshIEN(mesh), mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8, NULL);
	mesh->nodal_map_l2g_exterior = CdamTMalloc(index_type, mesh->num[0], HOST_MEM);
	/* Remove duplicate nodes and store in buffer */
	CountDistinctEntry(CdamMeshIEN(mesh), mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8, mesh->nodal_map_l2g_exterior);

	/* Adjust local node indices such that 
	 * if ownership differs, nodes owned by the current process have lower indices
	 * if both are owned locally, nodes with exclusive ownership have lower indices
	 * if both are ghosted, nodes with lower global indices have lower indices 
	 */
	index_type* node_is_ghosted = CdamTMalloc(index_type, smesh->num[0], HOST_MEM);
	CdamMemset(node_is_ghosted, 0, sizeof(index_type) * smesh->num[0], HOST_MEM);
	index_type  gid;
	for(i = 0; i < mesh->num[0]; i++) {
		gid = mesh->nodal_map_l2g_exterior[i];
		if(smesh->npart[gid] != mesh->rank) {
			node_is_ghosted[i] = 1;
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, node_is_ghosted, mesh->num[0], MPI_BYTE, MPI_MAX, mesh->comm);
	struct QsortCtx ctx = {node_is_ghosted, smesh->npart, mesh->rank};
	qsort_ctx = &ctx;
	qsort(mesh->nodal_map_l2g_exterior, mesh->num[0], sizeof(index_type), QsortCmpNpart);

	CdamFree(node_is_ghosted, sizeof(index_type) * smesh->num[0], HOST_MEM);

	/* Build the global to local map */
	index_type* g2l = CdamTMalloc(index_type, mesh->num[0] * 2, HOST_MEM);

	for(i = 0; i < mesh->num[0]; ++i) {
		g2l[i * 2 + 0] = mesh->nodal_map_l2g_exterior[i];
		g2l[i * 2 + 1] = i;
	}

	qsort(g2l, mesh->num[0], 2 * sizeof(index_type), QsortCmpIndexType);

	/* Update IEN by replacing the global node index with the local node index */
	index_type* p;
	for(i = 0; i < mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8; ++i) {
		p = bsearch(CdamMeshIEN(mesh) + i, g2l, mesh->num[0], 2 * sizeof(index_type), QsortCmpIndexType);
		if(p) {
			CdamMeshIEN(mesh)[i] = p[1];
		}
		else {
			fprintf(stderr, "Error: global node index %d not found.\n", CdamMeshIEN(mesh)[i]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	CdamFree(g2l, sizeof(index_type) * mesh->num[0] * 2, HOST_MEM);
}


static void GenerateL2GMapInterior(CdamMesh* mesh, SerialMesh* smesh) {
	int rank = mesh->rank;
	mesh->nodal_map_l2g_interior = CdamTMalloc(index_type, mesh->num[0], HOST_MEM);
	index_type* l2g_exterior = mesh->nodal_map_l2g_exterior;
	index_type* npart = smesh->npart;
	index_type num_node_owned = mesh->nodal_offset[rank + 1] - mesh->nodal_offset[rank];
	index_type i;
	/* Construction of nodal_map_l2g_interior consists of two parts: owned and ghosted nodes */
	/* Owned nodes are trivial */
	for(i = 0; i < num_node_owned; i++) {
		mesh->nodal_map_l2g_interior[i] = mesh->nodal_offset[rank] + i;
	}

	/* Ghosted nodes are tricky, since the map relies on the local indices of ghosted nodes 
	 * in their owner processors. */
	int* send_count = CdamTMalloc(int, mesh->num_procs, HOST_MEM);
	int* send_displ = CdamTMalloc(int, mesh->num_procs, HOST_MEM);
	int* recv_count = CdamTMalloc(int, mesh->num_procs, HOST_MEM);
	int* recv_displ = CdamTMalloc(int, mesh->num_procs, HOST_MEM);

	/* Count the ghosted nodes by ownership/partition */
	for(i = 0; i < mesh->num[0] - num_node_owned; ++i) {
		send_count[npart[l2g_exterior[i]]] += sizeof(index_type);
	}
	MPI_Alltoall(send_count, sizeof(int), MPI_CHAR,
							 recv_count, sizeof(int), MPI_CHAR, mesh->comm);

	for(i = 1; i < mesh->num_procs; ++i) {
		send_displ[i] = send_displ[i - 1] + send_count[i - 1]; 
		recv_displ[i] = recv_displ[i - 1] + recv_count[i - 1]; 
	}

	/* Send the global nodal indices to their owner processors to get the local nodal indices */
	int recv_buffer_size = recv_displ[mesh->num_procs - 1] + recv_count[mesh->num_procs - 1];
	index_type* node_index_buffer = CdamTMalloc(index_type, recv_buffer_size / sizeof(int), HOST_MEM);

	MPI_Alltoallv(l2g_exterior + num_node_owned, send_count, send_displ, MPI_CHAR,
								node_index_buffer, recv_count, recv_displ, MPI_CHAR, mesh->comm);

	/* Get the local nodal indices */
	void* p;
	for(i = 0; i < recv_buffer_size / sizeof(index_type); i++) {
		p = bsearch(node_index_buffer + i, l2g_exterior,
								mesh->num[0] - num_node_owned, sizeof(index_type),
								QsortCmpIndexType);

		node_index_buffer[i] = (index_type)(((index_type*)p) - l2g_exterior);
	}

	/* Fetch the local nodal indices of the ghosted nodes */
	MPI_Alltoallv(node_index_buffer, recv_count, recv_displ, MPI_CHAR,
								mesh->nodal_map_l2g_interior + num_node_owned,
								send_count, send_displ, MPI_CHAR, mesh->comm);

	index_type part;
	for(i = 0; i < mesh->num[0] - num_node_owned; ++i) {
		part = npart[mesh->nodal_map_l2g_interior[num_node_owned + i]];
		mesh->nodal_map_l2g_interior[num_node_owned + i] += mesh->nodal_offset[part];
	}


	CdamMemset(send_count, 0, sizeof(int) * mesh->num_procs, HOST_MEM);
	CdamMemset(send_displ, 0, sizeof(int) * mesh->num_procs, HOST_MEM);
	CdamMemset(recv_count, 0, sizeof(int) * mesh->num_procs, HOST_MEM);
	CdamMemset(recv_displ, 0, sizeof(int) * mesh->num_procs, HOST_MEM);

	CdamFree(send_count, mesh->num_procs * sizeof(int), HOST_MEM);
	CdamFree(send_displ, mesh->num_procs * sizeof(int), HOST_MEM);
	CdamFree(recv_count, mesh->num_procs * sizeof(int), HOST_MEM);
	CdamFree(recv_displ, mesh->num_procs * sizeof(int), HOST_MEM);
}

void CdamMeshLoad(CdamMesh* mesh, H5FileInfo* h5f, const char* group_name) {
	SerialMesh smesh;
	index_type* npart = NULL, *epart = NULL;
	ReadSerialMesh(h5f, group_name, &smesh); 

	MPI_Bcast(smesh.num, sizeof(index_type) * 4, MPI_CHAR, 0, mesh->comm);
	npart = CdamTMalloc(index_type, smesh.num[0], HOST_MEM);
	if(mesh->rank == 0) {
		epart = CdamTMalloc(index_type, smesh.num[1] + smesh.num[2] + smesh.num[3], HOST_MEM);
	}
	PartitionMesh(&smesh, epart, npart);
	MPI_Bcast(smesh.npart, sizeof(index_type) * smesh.num[0], MPI_CHAR, 0, MPI_COMM_WORLD);

	DistributeMesh(mesh, &smesh);
	GenerateL2GMapExterior(mesh, &smesh);
	GenerateL2GMapInterior(mesh, &smesh);

	/* Read nodes */
	CdamMeshCoord(mesh) = CdamTMalloc(value_type, mesh->num[0] * 3, HOST_MEM);
	LoadCoord(h5f, group_name, mesh->num[0], mesh->nodal_map_l2g_exterior, CdamMeshCoord(mesh));
}


void CdamMeshLoad_DO_NOT_USE_ME(CdamMesh* mesh, H5FileInfo* h5f, const char* group_name) {
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

	epart_count = CdamTMalloc(index_type, size * 3, HOST_MEM);
	CdamMemset(epart_count, 0, sizeof(index_type) * size * 3, HOST_MEM);
	if(rank == 0) {
		/* 0. Load element connectivity */
		LoadElementConnectivity(h5f, group_name, num, &ien);
		epart = CdamTMalloc(index_type, num[1] + num[2] + num[3], HOST_MEM);
		npart = CdamTMalloc(index_type, num[0], HOST_MEM);



		/* 1. Metis partition */
		PartitionMeshMetis(num, ien, size, epart, npart);

		FILE* fp = fopen("epart.txt", "w");
		for(i = 0; i < num[1] + num[2] + num[3]; ++i) {
			fprintf(fp, "%d\n", epart[i]);
		}
		fclose(fp);
		fp = fopen("npart.txt", "w");
		for(i = 0; i < num[0]; ++i) {
			fprintf(fp, "%d\n", npart[i]);
		}
		fclose(fp);

		/* 2. Shuffle the element connectivity */
		ShuffleIENByPartition(num, ien, epart);
		/* 2. Prepare the data for other processes */

		for(i = 0; i < num[1]; ++i) {
			epart_count[size * 0 + epart[i]]++;
		}
		for(i = 0; i < num[2]; ++i) {
			epart_count[size * 1 + epart[num[1] + i]]++;
		}
		for(i = 0; i < num[3]; ++i) {
			epart_count[size * 2 + epart[num[1] + num[2] + i]]++;
		}
	}

	/* Distribute the elements */
	MPI_Bcast(num, sizeof(index_type) * 4, MPI_CHAR, 0, mesh->comm);
	if(rank) {
		npart = CdamTMalloc(index_type, num[0], HOST_MEM);
	}
	MPI_Bcast(npart, sizeof(index_type) * num[0], MPI_CHAR, 0, mesh->comm);

	MPI_Bcast(epart_count, sizeof(index_type) * (size + 1) * 3, MPI_CHAR, 0, mesh->comm);

	mesh->num[0] = num[0];
	mesh->num[1] = epart_count[size * 0 + rank];
	mesh->num[2] = epart_count[size * 1 + rank];
	mesh->num[3] = epart_count[size * 2 + rank];

	CdamMeshIEN(mesh) = CdamTMalloc(index_type, mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8, HOST_MEM);
	
	int* send_count = CdamTMalloc(int, size, HOST_MEM);
	int* send_displ = CdamTMalloc(int, size, HOST_MEM);
	CdamMemset(send_count, 0, sizeof(int) * size, HOST_MEM);
	CdamMemset(send_displ, 0, sizeof(int) * size, HOST_MEM);

	for(i = 0; i < size; ++i) {
		send_count[i] = epart_count[size * 0 + i] * 4 * sizeof(index_type);
		if(i < size - 1) {
			send_displ[i + 1] = send_count[i] + send_displ[i];
		}
	}
	MPI_Scatterv(ien, send_count, send_displ, MPI_CHAR,
							 CdamMeshIEN(mesh), mesh->num[1] * 4 * sizeof(index_type), MPI_CHAR,
							 0, mesh->comm); 
	send_displ[0] = 0;
	for(i = 0; i < size; ++i) {
		send_count[i] = epart_count[size * 1 + i] * 6 * sizeof(index_type);
		if(i < size - 1) {
			send_displ[i + 1] = send_count[i] + send_displ[i];
		}
	}
	MPI_Scatterv(ien + num[1] * 4, send_count, send_displ, MPI_CHAR,
							 mesh->ien + mesh->num[1] * 4, mesh->num[2] * 6 * sizeof(index_type), MPI_CHAR,
							 0, mesh->comm);
	send_displ[0] = 0;
	for(i = 0; i < size; ++i) {
		send_count[i] = epart_count[size * 2 + i] * 8 * sizeof(index_type);
		if(i < size - 1) {
			send_displ[i + 1] = send_count[i] + send_displ[i];
		}
	}
	MPI_Scatterv(ien + num[1] * 4 + num[2] * 6, send_count, send_displ, MPI_CHAR,
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

	/* Generate the l2g_exterior */
	mesh->num[0] = CountDistinctEntry(CdamMeshIEN(mesh), mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8, NULL);
	if(0 && rank == 2) {
		FILE* fp = fopen("ien_local.txt", "w");
		for(i = 0; i < mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8; ++i) {
			fprintf(fp, "%d\n", mesh->ien[i]);
		}
		fclose(fp);

	}
	mesh->nodal_map_l2g_interior = CdamTMalloc(index_type, mesh->num[0], HOST_MEM);
	mesh->nodal_map_l2g_exterior = CdamTMalloc(index_type, mesh->num[0], HOST_MEM);
	CountDistinctEntry(CdamMeshIEN(mesh), mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8, mesh->nodal_map_l2g_exterior);

	int* npart_ownership = CdamTMalloc(int, num[0], HOST_MEM);
	CdamMemset(npart_ownership, 0, sizeof(int) * num[0], HOST_MEM);
	for(i = 0; i < mesh->nodal_offset[rank + 1] - mesh->nodal_offset[rank]; ++i) {
		/* mark the nodes owned by other processes */
		if(npart[mesh->nodal_map_l2g_exterior[i]] != rank) {
			npart_ownership[mesh->nodal_map_l2g_exterior[i]] = 1;
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, npart_ownership, num[0], MPI_INT, MPI_MAX, mesh->comm);
	qsort_ctx = &(struct QsortCtx){npart, npart_ownership, size};
	qsort(mesh->nodal_map_l2g_exterior, mesh->num[0], sizeof(index_type), QsortCmpNpart);
	mesh->num_exclusive_node = 0;
	for(i = 0; i < mesh->nodal_offset[rank + 1] - mesh->nodal_offset[rank]; ++i) {
		if(npart_ownership[mesh->nodal_map_l2g_exterior[i]] == 0) {
			mesh->num_exclusive_node++;
		}
	}
	CdamFree(npart_ownership, sizeof(int) * num[0], HOST_MEM);
	/* Update mesh->ien using mesh->nodal_map_l2g_exterior */
	/* Generate a g2l map */
	index_type* g2l = CdamTMalloc(index_type, mesh->num[0] * 2, HOST_MEM);
	for(i = 0; i < mesh->num[0]; ++i) {
		g2l[i * 2 + 0] = mesh->nodal_map_l2g_exterior[i];
		g2l[i * 2 + 1] = i;
	}

	qsort(g2l, mesh->num[0], sizeof(index_type) * 2, QsortCmpIndexType);
	for(i = 0; i < mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8; ++i) {
		index_type* p = (index_type*)bsearch(mesh->ien + i, g2l, mesh->num[0], sizeof(index_type) * 2, QsortCmpIndexType);
		mesh->ien[i] = p[0];
	}

	printf("rank = %d, num_exclusive_node = %d\n", rank, mesh->num_exclusive_node);
	printf("rank = %d, num_owned_node = %d\n", rank, mesh->nodal_offset[rank + 1] - mesh->nodal_offset[rank]);
	printf("rank = %d, num = %d, %d, %d, %d\n", rank, mesh->num[0], mesh->num[1], mesh->num[2], mesh->num[3]);
	return;


	/* Generate the l2g_interior */
	int* send_node_count = CdamTMalloc(int, size * 2 + 1, HOST_MEM);
	int* send_node_offset = send_node_count + size;
	int* recv_node_count = CdamTMalloc(int, size * 2 + 1, HOST_MEM);
	int* recv_node_offset = recv_node_count + size;

	CdamMemset(send_node_count, 0, sizeof(int) * (size * 2 + 1), HOST_MEM);
	CdamMemset(recv_node_count, 0, sizeof(int) * (size * 2 + 1), HOST_MEM);
	index_type num_owned_node = mesh->nodal_offset[rank + 1] - mesh->nodal_offset[rank];
	for(i = 0; i < mesh->num[0] - num_owned_node; ++i) {
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
																				 mesh->num[0] - num_owned_node, sizeof(index_type),
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
	for(i = 0; i < mesh->num[0] - num_owned_node; ++i) {
		index_type part = npart[mesh->nodal_map_l2g_interior[i + num_owned_node]];
		mesh->nodal_map_l2g_interior[i + num_owned_node] += mesh->nodal_offset[part];
	}

	CdamMeshCoord(mesh) = CdamTMalloc(value_type, mesh->num[0] * 3, HOST_MEM);
	snprintf(dataset_name, sizeof(dataset_name) / sizeof(char), "%s/xg", group_name);
	index_type* hindex = CdamTMalloc(index_type, mesh->num[0] * 3, HOST_MEM);
	for(i = 0; i < mesh->num[0]; ++i) {
		hindex[i * 3 + 0] = mesh->nodal_map_l2g_interior[i] * 3 + 0;
		hindex[i * 3 + 1] = mesh->nodal_map_l2g_interior[i] * 3 + 1;
		hindex[i * 3 + 2] = mesh->nodal_map_l2g_interior[i] * 3 + 2;
	}
	H5ReadDatasetValIndexed(h5f, dataset_name, mesh->num[0],
													hindex, CdamMeshCoord(mesh));

	CdamFree(hindex, sizeof(index_type) * mesh->num[0] * 3, HOST_MEM);
	CdamFree(epart_count, sizeof(index_type) * size * 3, HOST_MEM);
	CdamFree(epart, sizeof(index_type) * (num[1] + num[2] + num[3]), HOST_MEM);
	CdamFree(npart, sizeof(index_type) * num[0], HOST_MEM);
	CdamFree(ien, sizeof(index_type) * (num[1] + num[2] + num[3]), HOST_MEM);
	CdamFree(send_count, sizeof(int) * size, HOST_MEM);
	CdamFree(send_displ, sizeof(int) * size, HOST_MEM);
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

void CdamMeshPrefetch(CdamMesh *mesh) {
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
	index_type max_color = GetMaxColor(mesh->color, num_elem, scratch);
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

