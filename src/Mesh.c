
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "alloc.h"
#include "h5util.h"
#include "MeshData.h"
#include "Mesh.h"


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

