
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "alloc.h"
#include "h5util.h"
#include "MeshData.h"
#include "Mesh.h"


__BEGIN_DECLS__
static void ReadBoundFromH5Private(Mesh3D* mesh, H5FileInfo* h5f, const char* group_name) {
	char dataset_name[256];
	index_type num_facet, num_bnode;
	index_type buffer_size = 0;
	index_type* buffer = NULL;
	/* Read num of bound */
	sprintf(dataset_name, "%s/bound/node_offset", group_name);
	H5GetDatasetSize(h5f, dataset_name, &mesh->num_bound);
	mesh->num_bound--;
	mesh->bound_node_offset = (index_type*)CdamMallocHost(SIZE_OF(index_type) * (mesh->num_bound + 1) * 2);
	mesh->bound_elem_offset = mesh->bound_node_offset + mesh->num_bound + 1;

	/* Read bound node offset */
	sprintf(dataset_name, "%s/bound/node_offset", group_name);
	H5ReadDatasetInd(h5f, dataset_name, mesh->bound_node_offset);

	/* Read bound elem offset */
	sprintf(dataset_name, "%s/bound/elem_offset", group_name);
	H5ReadDatasetInd(h5f, dataset_name, mesh->bound_elem_offset);

	/* Allocate the buffer for later readding */
	num_bnode = mesh->bound_node_offset[mesh->num_bound];
	num_facet = mesh->bound_elem_offset[mesh->num_bound];
	buffer_size = (num_bnode > num_facet) ? num_bnode : num_facet;
	buffer = (index_type*)CdamMallocHost(SIZE_OF(index_type) * buffer_size);

	/* Allocate the memory for bound node and bound facet to edge */
	mesh->bound_node = (index_type*)CdamMallocDevice(SIZE_OF(index_type) * (num_bnode + num_facet * 2));
	mesh->bound_f2e = mesh->bound_node + num_bnode;
	mesh->bound_forn = mesh->bound_f2e + num_facet;

	/* Read bound node */
	sprintf(dataset_name, "%s/bound/node", group_name);
	H5ReadDatasetInd(h5f, dataset_name, buffer);
	cudaMemcpy(mesh->bound_node, buffer, SIZE_OF(index_type) * num_bnode, cudaMemcpyHostToDevice);
	
	/* Read bound f2e */
	sprintf(dataset_name, "%s/bound/f2e", group_name);
	H5ReadDatasetInd(h5f, dataset_name, buffer);
	cudaMemcpy(mesh->bound_f2e, buffer, SIZE_OF(index_type) * num_facet, cudaMemcpyHostToDevice);

	/* Read bound facet orientation */
	sprintf(dataset_name, "%s/bound/forn", group_name);
	H5ReadDatasetInd(h5f, dataset_name, buffer);
	cudaMemcpy(mesh->bound_forn, buffer, SIZE_OF(index_type) * num_facet, cudaMemcpyHostToDevice);

	CdamFreeHost(buffer, SIZE_OF(index_type) * buffer_size);
}

Mesh3D* Mesh3DCreate(index_type num_node, index_type num_tet, index_type num_prism, index_type num_hex) {
	Mesh3D* mesh = (Mesh3D*)CdamMallocHost(SIZE_OF(Mesh3D));
	ASSERT(num_node >= 4 && "Invalid number of nodes");
	ASSERT(num_tet + num_prism + num_hex && "Invalid number of elements");
	memset(mesh, 0, SIZE_OF(Mesh3D));
	Mesh3DNumNode(mesh) = num_node;
	Mesh3DNumTet(mesh) = num_tet;
	Mesh3DNumPrism(mesh) = num_prism;
	Mesh3DNumHex(mesh) = num_hex;

	Mesh3DHost(mesh) = Mesh3DDataCreateHost(num_node, num_tet, num_prism, num_hex);
	Mesh3DDevice(mesh) = Mesh3DDataCreateDevice(num_node, num_tet, num_prism, num_hex);


	return mesh;
}

Mesh3D* Mesh3DCreateH5(H5FileInfo* h5f, const char* group_name) {
	Mesh3D* mesh;
	index_type num_node, num_tet, num_prism, num_hex;
	ASSERT(h5f && "Invalid file");
	ASSERT(group_name && "Invalid group name");

	mesh = (Mesh3D*)CdamMallocHost(SIZE_OF(Mesh3D));
	memset(mesh, 0, SIZE_OF(Mesh3D));

	Mesh3DHost(mesh) = Mesh3DDataCreateH5(h5f, group_name);
	num_node = Mesh3DDataNumNode(Mesh3DHost(mesh));
	num_tet = Mesh3DDataNumTet(Mesh3DHost(mesh));
	num_prism = Mesh3DDataNumPrism(Mesh3DHost(mesh));
	num_hex = Mesh3DDataNumHex(Mesh3DHost(mesh));

	Mesh3DDevice(mesh) = Mesh3DDataCreateDevice(num_node, num_tet, num_prism, num_hex);
	Mesh3DDataCopy(Mesh3DDevice(mesh), Mesh3DHost(mesh), H2D);
	
	Mesh3DNumNode(mesh) = num_node;
	Mesh3DNumTet(mesh) = num_tet;
	Mesh3DNumPrism(mesh) = num_prism;
	Mesh3DNumHex(mesh) = num_hex;

	ReadBoundFromH5Private(mesh, h5f, group_name);

	return mesh;
}

void Mesh3DDestroy(Mesh3D* mesh) {
	index_type num_elem = Mesh3DNumTet(mesh) + Mesh3DNumPrism(mesh) + Mesh3DNumHex(mesh);

	index_type num_bnode = mesh->bound_node_offset[mesh->num_bound];
	index_type num_facet = mesh->bound_elem_offset[mesh->num_bound];

	Mesh3DDataDestroy(Mesh3DHost(mesh));
	Mesh3DDataDestroy(Mesh3DDevice(mesh));
	// CdamFreeDevice(mesh->epart, SIZE_OF(index_type) * num_elem);

	/* Free bound data */
	CdamFreeHost(mesh->bound_node_offset, SIZE_OF(index_type) * (mesh->num_bound + 1) * 2);
	CdamFreeDevice(mesh->bound_node, SIZE_OF(index_type) * (num_bnode + num_facet * 2));

	/* Free batch data */
	CdamFreeHost(mesh->batch_offset, SIZE_OF(index_type) * (mesh->num_batch + 1));
	CdamFreeDevice(mesh->batch_ind, SIZE_OF(index_type) * num_elem);

	/* Free color data */
	CdamFreeDevice(mesh->color, SIZE_OF(color_t) * num_elem);
	CdamFreeHost(mesh, SIZE_OF(Mesh3D));
}



void Mesh3DUpdateHost(Mesh3D* mesh) {
	ASSERT(mesh && "Invalid mesh");
	ASSERT(Mesh3DHost(mesh) && "Host memory is not allocated");
	ASSERT(Mesh3DDevice(mesh) && "Device memory is not allocated");

	Mesh3DDataCopy(Mesh3DHost(mesh), Mesh3DDevice(mesh), D2H);
}

void Mesh3DUpdateDevice(Mesh3D* mesh) {
	ASSERT(mesh && "Invalid mesh");
	ASSERT(Mesh3DHost(mesh) && "Host memory is not allocated");
	ASSERT(Mesh3DDevice(mesh) && "Device memory is not allocated");

	Mesh3DDataCopy(Mesh3DDevice(mesh), Mesh3DHost(mesh), H2D);
}

// #include "partition.h"
// void Mesh3DPartition(Mesh3D* mesh, index_type num_part) {
// 	index_type num_elem = Mesh3DNumTet(mesh) + Mesh3DNumPrism(mesh) + Mesh3DNumHex(mesh);
// 	mesh->num_part = num_part;
// 	mesh->epart = CdamMallocDevice(SIZE_OF(index_type) * num_elem);
// 	PartitionMesh3DMETIS(mesh, num_part);
// }

void Mesh3DColor(Mesh3D* mesh) {
	index_type num_elem = Mesh3DNumTet(mesh) + Mesh3DNumPrism(mesh) + Mesh3DNumHex(mesh);
	mesh->color = CdamMallocDevice(SIZE_OF(color_t) * num_elem);
	ColorMeshTet(mesh, MAX_COLOR, mesh->color);
	mesh->num_color = GetMaxColor(mesh->color, Mesh3DNumTet(mesh)) + 1;
}

index_type CountValueColorLegacy(color_t*, index_type, color_t);
void FindValueColor(color_t*, index_type, color_t, index_type*);

void Mesh3DGenerateColorBatch(Mesh3D* mesh) {
	index_type num_tet = Mesh3DNumTet(mesh);
	index_type num_prism = Mesh3DNumPrism(mesh);
	index_type num_hex = Mesh3DNumHex(mesh);
	index_type num_elem = num_tet + num_prism + num_hex;
	mesh->color = (color_t*)CdamMallocDevice(SIZE_OF(color_t) * num_elem);
	mesh->batch_ind = (index_type*)CdamMallocDevice(SIZE_OF(index_type) * num_elem);

	if(num_tet) {
		
		index_type batch_size = 0;
		index_type max_batch_size = 1000;

		ColorMeshTet(mesh, MAX_COLOR, mesh->color);
		index_type num_color = GetMaxColor(mesh->color, num_tet) + 1;
		mesh->num_color = num_color;

		mesh->num_batch = num_color;
		mesh->batch_offset = (index_type*)CdamMallocHost(SIZE_OF(index_type) * (num_color + 1));
		mesh->batch_offset[0] = 0;

		for(index_type c = 0; c < num_color; c++) {
			batch_size = CountValueColorLegacy(mesh->color, num_tet, c);
			mesh->batch_offset[c + 1] = mesh->batch_offset[c] + batch_size;
			if(batch_size > max_batch_size) {
				max_batch_size = batch_size;
			}
		}

		index_type* batch = (index_type*)CdamMallocDevice(SIZE_OF(index_type) * max_batch_size);

		for(index_type c = 0; c < num_color; c++) {
			FindValueColor(mesh->color, num_tet, c, batch);
			cudaMemcpy(mesh->batch_ind + mesh->batch_offset[c],
								 batch,
								 SIZE_OF(index_type) * (mesh->batch_offset[c + 1] - mesh->batch_offset[c]),
								 cudaMemcpyDeviceToDevice);
		}	
		CdamFreeDevice(batch, SIZE_OF(index_type) * max_batch_size);
	}

}

void CDAM_MeshCreate(MPI_Comm comm, CDAM_Mesh** mesh) {
	*mesh = (CDAM_Mesh*)CdamMallocHost(SIZE_OF(CDAM_Mesh));
	memset(*mesh, 0, SIZE_OF(CDAM_Mesh));
	(*mesh)->comm = comm;
	MPI_Comm_rank(comm, &(mesh[0]->rank));
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
	(*mesh)->mem_location = HYPRE_MEMORY_DEVICE;
#else
	(*mesh)->mem_location = HYPRE_MEMORY_HOST;
#endif
}

void CDAM_MeshDestroy(CDAM_Mesh* mesh) {
	index_type num_elem = 0;
	index_type num_facet = 0;
	num_elem = 0;
	num_elem += CDAM_MeshNumTet(mesh) * 4;
	num_elem += CDAM_MeshNumPrism(mesh) * 6;
	num_elem += CDAM_MeshNumHex(mesh) * 8;
	num_facet = mesh->bound_offset[CDAM_MeshNumBound(mesh)];
	if(mesh->mem_location == HYPRE_MEMORY_DEVICE) {
		CdamFreeDevice(CDAM_MeshCoord(mesh), SIZE_OF(value_type) * CDAM_MeshNumNode(mesh) * 3);
		CdamFreeDevice(CDAM_MeshElem(mesh), SIZE_OF(index_type) * tmp);
		CdamFreeDevice(mesh->bound_id, SIZE_OF(index_type) * CDAM_MeshNumBound(mesh));
		CdamFreeDevice(mesh->bound_offset, SIZE_OF(index_type) * (CDAM_MeshNumBound(mesh) + 1));

		CdamFreeDevice(mesh->bound_f2e, SIZE_OF(index_type) * tmp);
		CdamFreeDevice(mesh->bound_forn, SIZE_OF(index_type) * tmp);

		CdamFreeDevice(mesh->nodal_offset, SIZE_OF(index_type) * (mesh->num_part + 1));

		CdamFreeDevice(mesh->nodal_map_l2g, SIZE_OF(index_type) * CDAM_MeshNumNode(mesh));
		CdamFreeDevice(mesh->elem_map_l2g, SIZE_OF(index_type) * num_elem);

		CdamFreeDevice(mesh->color, SIZE_OF(index_type) * num_elem);
		CdamFreeDevice(mesh->batch_offset, SIZE_OF(index_type) * (mesh->num_batch + 1));
		CdamFreeDevice(mesh->batch_ind, SIZE_OF(index_type) * num_elem);
	}
	else {
		CdamFreeHost(CDAM_MeshCoord(mesh), SIZE_OF(value_type) * CDAM_MeshNumNode(mesh) * 3);
		CdamFreeHost(CDAM_MeshElem(mesh), SIZE_OF(index_type) * tmp);
		CdamFreeHost(mesh->bound_id, SIZE_OF(index_type) * CDAM_MeshNumBound(mesh));
		CdamFreeHost(mesh->bound_offset, SIZE_OF(index_type) * (CDAM_MeshNumBound(mesh) + 1));

		CdamFreeHost(mesh->bound_f2e, SIZE_OF(index_type) * tmp);
		CdamFreeHost(mesh->bound_forn, SIZE_OF(index_type) * tmp);

		CdamFreeHost(mesh->nodal_offset, SIZE_OF(index_type) * (mesh->num_part + 1));

		CdamFreeHost(mesh->nodal_map_l2g, SIZE_OF(index_type) * CDAM_MeshNumNode(mesh));
		CdamFreeHost(mesh->elem_map_l2g, SIZE_OF(index_type) * num_elem);

		CdamFreeHost(mesh->color, SIZE_OF(index_type) * num_elem);
		CdamFreeHost(mesh->batch_offset, SIZE_OF(index_type) * (mesh->num_batch + 1));
		CdamFreeHost(mesh->batch_ind, SIZE_OF(index_type) * num_elem);

	}
	CdamFreeHost(mesh, SIZE_OF(CDAM_Mesh));
}

static void LoadElementConnectivity(H5FileInfo* h5f, const char* group_name, index_type num[], index_type** ien) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	char dataset_name[256];
	ASERT(strlen(group_name) < 192 && "Invalid group name: too long (>= 192).\n");

	/* Read the number of nodes */
	snprintf(dataset_name, SIZE_OF(data_name) / SIZE_OF(char), "%s/xg", group_name);
	H5GetDatasetSize(h5f, dataset_name, num);
	num[0] = num[0] / 3;

	/* Read the number of elements */
	snprintf(dataset_name, SIZE_OF(data_name) / SIZE_OF(char), "%s/ien/tet", group_name);
	H5GetDatasetSize(h5f, dataset_name, num + 1);

	snprintf(dataset_name, SIZE_OF(data_name) / SIZE_OF(char), "%s/ien/prism", group_name);
	H5GetDatasetSize(h5f, dataset_name, num + 2);

	snprintf(dataset_name, SIZE_OF(data_name) / SIZE_OF(char), "%s/ien/hex", group_name);
	H5GetDatasetSize(h5f, dataset_name, num + 3);

	/* Allocate memory for element connectivity */
	index_type* eind = (index_type*)CdamMallocHost(SIZE_OF(index_type) * (num[1] + num[2] + num[3]));
	num[1] /= 4; 
	num[2] /= 6;
	num[3] /= 8;

	snprintf(dataset_name, SIZE_OF(data_name) / SIZE_OF(char), "%s/ien/tet", group_name);
	H5ReadDatasetInd(h5f, dataset_name, eind);

	snprintf(dataset_name, SIZE_OF(data_name) / SIZE_OF(char), "%s/ien/prism", group_name);
	H5ReadDatasetInd(h5f, dataset_name, eind + num[1] * 4);

	snprintf(dataset_name, SIZE_OF(data_name) / SIZE_OF(char), "%s/ien/hex", group_name);
	H5ReadDatasetInd(h5f, dataset_name, eind + num[1] * 4 + num[2] * 6);

	*ien = eind;
}

struct cdam_mesh_part_struct {
	index_type num_node_owned;
	index_type num_node_ghosted;

	index_type* node_map_l2g;

	index_type num_elem_owned;
	index_type* elem_map_l2g;
};
typedef struct cdam_mesh_part_struct CDAM_MeshPart;

struct QsortCtx {
	index_type* ien;
	index_type* epart;
	index_type size;
};


static void* qsort_ctx = NULL;
static int QsortCmpElem(const void* a, const void* b) {
	index_type* ien = ((struct QsortCtx*)qsort_ctx)->ien;
	index_type* epart = ((struct QsortCtx*)qsort_ctx)->epart;
	index_type size = ((struct QsortCtx*)qsort_ctx)->size;

	index_type ia = ((index_type*)a - ien) / size;
	index_type ib = ((index_type*)b - ien) / size;

	return epart[ia] - epart[ib];
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
	return *((index_type*)a) - *((index_type*)b);
}
static index_type CountDistinctEntry(index_type* array, index_type size, index_type* array_set) {
	index_type i, j;
	index_type count = 0;
	index_type* array_sorted = (index_type*)CdamMallocHost(SIZE_OF(index_type) * size);
	memcpy(array_sorted, array, SIZE_OF(index_type) * size);
	qsort(array_sorted, size, SIZE_OF(index_type), QsortCmpIndexType);

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
	index_type* npart = (index_type*)((struct QsortCtx*)qsort_ctx)->npart;
	index_type rank = ((struct QsortCtx*)qsort_ctx)->size;
	index_type ia = *(index_type*)a;
	index_type ib = *(index_type*)b;
	index_type owned_a = npart[ia] == rank;
	index_type owned_b = npart[ib] == rank;

	if(owned_a && !owned_b) {
		return -1;
	}
	else if(!owned_a && owned_b) {
		return 1;
	}
	else {
		return npart[ia] - npart[ib];
	}
}

void CDAM_MeshLoad(CDAM_Mesh* mesh, H5FileInfo* h5f, const char* group_name) {
	index_type i;
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	index_type* count_epart = NULL;
	index_type* count_npart = NULL;
	index_type num[4];
	index_type* ien = NULL;
	index_type* epart = NULL;
	index_type* npart = NULL;

	count_epart = (index_type*)CdamMallocHost(SIZE_OF(index_type) * (size + 1) * 3);
	memset(count_epart, 0, SIZE_OF(index_type) * (size + 1) * 3);
	if(rank == 0) {
		/* 0. Load element connectivity */
		LoadElementConnectivity(h5f, group_name, num, &ien);
		epart = (index_type*)CdamMallocHost(SIZE_OF(index_type) * (num[1] + num[2] + num[3]));
		npart = (index_type*)CdamMallocHost(SIZE_OF(index_type) * num[0]);

		/* 1. Metis partition */
		PartitionMeshMetis(num, ien, size, epart, npart);

		/* 2. Shuffle the element connectivity */
		ShuffleIENByPartition(num, ien, epart);
		/* 2. Prepare the data for other processes */

		for(i = 0; i < num[1]; ++i) {
			count_epart[size * 0 + epart[i] + 1]++;
		}
		for(i = 0; i < num[2]; ++i) {
			count_epart[size * 1 + epart[num[1] + i] + 1]++;
		}
		for(i = 0; i < num[3]; ++i) {
			count_epart[size * 2 + epart[num[1] + num[2] + i] + 1]++;
		}
		for(i = 0; i < size; ++i) {
			count_epart[size * 0 + i + 1] += count_epart[size * 0 + i];
			count_epart[size * 1 + i + 1] += count_epart[size * 1 + i];
			count_epart[size * 2 + i + 1] += count_epart[size * 2 + i];
		}
	}

	MPI_Bcast(num, SIZE_OF(index_type) * 4, MPI_CHAR, 0, mesh->comm);
	if(rank) {
		npart = (index_type*)CdamMallocHost(SIZE_OF(index_type) * num[0]);
	}
	MPI_Bcast(npart, SIZE_OF(index_type) * num[0], MPI_CHAR, 0, mesh->comm);

	MPI_Bcast(count_epart, SIZE_OF(index_type) * (size + 1) * 3, MPI_CHAR, 0, mesh->comm);

	mesh->num[0] = num[0];
	mesh->num[1] = count_epart[size * 0 + rank + 1] - count_epart[size * 0 + rank];
	mesh->num[2] = count_epart[size * 1 + rank + 1] - count_epart[size * 1 + rank];
	mesh->num[3] = count_epart[size * 2 + rank + 1] - count_epart[size * 2 + rank];

	mesh->ien = (index_type*)CdamMallocHost(SIZE_OF(index_type) 
			* (mesh->num[1] * 4 + mesh->num[2] * 6 + mesh->num[3] * 8));
	
	MPI_Count* send_count = (MPI_Count*)CdamMallocHost(SIZE_OF(MPI_Count) * size);
	MPI_Aint* send_disp = (MPI_Aint*)CdamMallocHost(SIZE_OF(MPI_Aint) * size);

	for(i = 0; i < size; ++i) {
		send_count[i] = count_epart[size * 0 + i + 1] - count_epart[size * 0 + i];
		send_disp[i] = count_epart[size * 0 + i] * 4 * SIZE_OF(index_type);
	}
	MPI_Scatterv(ien, send_count, send_disp, MPI_CHAR,
							 mesh->ien, mesh->num[1] * 4 * SIZE_OF(index_type), MPI_CHAR,
							 0, mesh->comm); 
	for(i = 0; i < size; ++i) {
		send_count[i] = count_epart[size * 1 + i + 1] - count_epart[size * 1 + i];
		send_disp[i] = count_epart[size * 1 + i] * 6 * SIZE_OF(index_type);
	}
	MPI_Scatterv(ien + num[1] * 4, send_count, send_disp, MPI_CHAR,
							 mesh->ien + mesh->num[1] * 4, mesh->num[2] * 6 * SIZE_OF(index_type), MPI_CHAR,
							 0, mesh->comm);
	for(i = 0; i < size; ++i) {
		send_count[i] = count_epart[size * 2 + i + 1] - count_epart[size * 2 + i];
		send_disp[i] = count_epart[size * 2 + i] * 8 * SIZE_OF(index_type);
	}
	MPI_Scatterv(ien + num[1] * 4 + num[2] * 6, send_count, send_disp, MPI_CHAR,
							 mesh->ien + mesh->num[1] * 4 + mesh->num[2] * 6, mesh->num[3] * 8 * SIZE_OF(index_type), MPI_CHAR,
							 0, mesh->comm);


	mesh->rank = rank;
	mesh->num_part = size;

	mesh->nodal_offset = (index_type*)CdamMallocHost(SIZE_OF(index_type) * (size + 1));
	memset(mesh->nodal_offset, 0, SIZE_OF(index_type) * (size + 1));

	for(i = 0; i < num[0]; ++i) {
		mesh->nodal_offset[npart[i] + 1]++;
	}
	for(i = 0; i < size; ++i) {
		mesh->nodal_offset[i + 1] += mesh->nodal_offset[i];
	}

	mesh->num[0] = CountDistinctEntry(mesh->ien, mesh->num[0] * 4 + mesh->num[1] * 6 + mesh->num[2] * 8, NULL);
	mesh->nodal_map_l2g_interior = (index_type*)CdamMallocHost(SIZE_OF(index_type) * mesh->num[0]);
	mesh->nodal_map_l2g_exterior = (index_type*)CdamMallocHost(SIZE_OF(index_type) * mesh->num[0]);
	CountDistinctEntry(mesh->ien, mesh->num[0] * 4 + mesh->num[1] * 6 + mesh->num[2] * 8, mesh->nodal_map_l2g_exterior);

	qsort_ctx = &(struct QsortCtx){npart, NULL, size};
	QsortCmpNpart(mesh->nodal_map_l2g_exterior, mesh->num[0], SIZE_OF(index_type), QsortCmpNpart);


	/* Generate the l2g_interior */
	int* send_node_count = (int*)CdamMallocHost(SIZE_OF(int) * (size * 2 + 1));
	int* send_node_offset = send_node_count + size + 1;
	int* recv_node_count = (int*)CdamMallocHost(SIZE_OF(int) * (size * 2 + 1));
	int* recv_node_offset = recv_node_count + size + 1;

	memset(send_node_count, 0, SIZE_OF(int) * (size * 2 + 1));
	memset(recv_node_count, 0, SIZE_OF(int) * (size * 2 + 1));
	index_type num_owned_node = mesh->nodal_offset[rank + 1] - mesh->nodal_offset[rank];
	for(i = 0; i < num[0] - num_owned_node; ++i) {
		send_node_count[npart[mesh->nodal_map_l2g_exterior[i + num_owned_node]]] += SIZE_OF(int);
	}
	MPI_Alltoall(send_node_count, SIZE_OF(int), MPI_CHAR,
							 recv_node_count, SIZE_OF(int), MPI_CHAR, mesh->comm);
	for(i = 0; i < size; ++i) {
		send_node_offset[i + 1] = send_node_count[i] + send_node_offset[i];
		recv_node_offset[i + 1] = recv_node_count[i] + recv_node_offset[i];
	}
	index_type* recv_buff = (index_type*)CdamMallocHost(SIZE_OF(index_type) * recv_node_offset[size]);
	MPI_Alltoallv(mesh->nodal_map_l2g_exterior + num_owned_node, send_node_count, send_node_offset, MPI_CHAR,
							 recv_buff, recv_node_count, recv_node_offset, MPI_CHAR, mesh->comm);

	for(i = 0; i < recv_node_offset[size]; ++i) {
		/* find the local index in mesh->nodal_map_l2g_exterior */
		index_type* p = (index_type*)bsearch(recv_buff + i, mesh->nodal_map_l2g_exterior,
																				 num[0] - num_owned_node, SIZE_OF(index_type),
																				 QsortCmpIndexType);
		recv_buff[i] = p - mesh->nodal_map_l2g_exterior;
	}
	MPI_Alltoallv(recv_buff, recv_node_count,
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
	CdamFreeHost(recv_buff, SIZE_OF(index_type) * recv_node_offset[size]);
	CdamFreeHost(send_node_count, SIZE_OF(int) * (size * 2 + 1));
	CdamFreeHost(recv_node_count, SIZE_OF(int) * (size * 2 + 1));

	
}

static void CountPartitionNode(index_type num[], index_type* ien,
																	index_type* npart, index_type* epart,
																	index_type* count) {
	int rank, size;
	index_type i, j;
	index_type num_node = num[0];
	index_type num_tet = num[1], num_prism = num[2], num_hex = num[3];

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	memset(count, 0, SIZE_OF(index_type) * size * 2);

	/* Count the number of owned nodes */
	for(i = 0; i < num_node; ++i) {
		count[npart[i]]++;
	}

	/* Count the number of ghosted nodes */
	for(i = 0; i < num_tet; ++i) {
		j = epart[i];
		count[j + size] += epart[ien[i * 4 + 0]] != j; 
		count[j + size] += epart[ien[i * 4 + 1]] != j; 
		count[j + size] += epart[ien[i * 4 + 2]] != j; 
		count[j + size] += epart[ien[i * 4 + 3]] != j; 
	}
	for(i = 0; i < num_prism; ++i) {
		j = epart[num_tet + i];
		count[j + size] += epart[ien[(num_tet + i) * 6 + 0]] != j;
		count[j + size] += epart[ien[(num_tet + i) * 6 + 1]] != j;
		count[j + size] += epart[ien[(num_tet + i) * 6 + 2]] != j;
		count[j + size] += epart[ien[(num_tet + i) * 6 + 3]] != j;
		count[j + size] += epart[ien[(num_tet + i) * 6 + 4]] != j;
		count[j + size] += epart[ien[(num_tet + i) * 6 + 5]] != j;
	}
	for(i = 0; i < num_hex; ++i) {
		j = epart[num_tet + num_prism + i];
		count[j + size] += epart[ien[(num_tet + num_prism + i) * 8 + 0]] != j;
		count[j + size] += epart[ien[(num_tet + num_prism + i) * 8 + 1]] != j;
		count[j + size] += epart[ien[(num_tet + num_prism + i) * 8 + 2]] != j;
		count[j + size] += epart[ien[(num_tet + num_prism + i) * 8 + 3]] != j;
		count[j + size] += epart[ien[(num_tet + num_prism + i) * 8 + 4]] != j;
		count[j + size] += epart[ien[(num_tet + num_prism + i) * 8 + 5]] != j;
		count[j + size] += epart[ien[(num_tet + num_prism + i) * 8 + 6]] != j;
		count[j + size] += epart[ien[(num_tet + num_prism + i) * 8 + 7]] != j;
	}
}

static void GetPartitionNode(index_type num[], index_type* ien,
															index_type* npart, index_type* epart,
															index_type* count, index_type* owned, index_type* ghosted) {
	int rank, size;
	index_type i, j;
	index_type num_node = num[0];
	index_type num_tet = num[1], num_prism = num[2], num_hex = num[3];
	index_type* owned_offset, *ghosted_offset;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	owned_offset = (index_type*)CdamMallocHost(SIZE_OF(index_type) * (size + 1));
	ghosted_offset = (index_type*)CdamMallocHost(SIZE_OF(index_type) * (size + 1));
	owned_offset[0] = 0;
	memcpy(owned_offset + 1, count, SIZE_OF(index_type) * size);
	ghosted_offset[0] = 0;
	memcpy(ghosted_offset + 1, count + size, SIZE_OF(index_type) * size);
	for(i = 0; i < size; ++i) {
		owned_offset[i + 1] += owned_offset[i];
		ghosted_offset[i + 1] += ghosted_offset[i];
	}

	/* Get the owned nodes */
	for(i = 0; i < num_node; ++i) {
		owned[owned_offset[npart[i]]++] = i;
	}

	/* Get the ghosted nodes */


}

static void ReorderNode(index_type num_node,
												index_type* npart, index_type* count,
												index_type* map, index_type* rmap) {
	int rank, size;
	index_type i, j;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* 1. Reorder the node according to the partition in the contiguous way: 
	 * part j contains v[count[j] : count[j+1]] */
	for(i = 0; i < num_node; ++i) {
		map[count[npart[i]]++] = i;
	}
	memmove(count + 1, count, SIZE_OF(index_type) * size);
	count[0] = 0;

	/* 2. Reverse the order of the node map */
	for(i = 0; i < num_node; ++i) {
		rmap[map[i]] = i;
	}
}


void CDAM_MeshToDevice(CDAM_Mesh *mesh) {
	index_type num_node = mesh->num[0];
	index_type num_tet = mesh->num[1];
	index_type num_prism = mesh->num[2];
	index_type num_hex = mesh->num[3];

	value_type* coord = (value_type*)CdamMallocDevice(SIZE_OF(value_type) * num_node * 3);

}





__END_DECLS__

