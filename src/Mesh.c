
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "alloc.h"
#include "h5util.h"
#include "MeshData.h"
#include "Mesh.h"


Mesh3D* Mesh3DCreate(u32 num_node, u32 num_tet, u32 num_prism, u32 num_hex) {
	Mesh3D* mesh = (Mesh3D*)CdamMallocHost(sizeof(Mesh3D));
	ASSERT(num_node >= 4 && "Invalid number of nodes");
	ASSERT(num_tet + num_prism + num_hex && "Invalid number of elements");
	memset(mesh, 0, sizeof(Mesh3D));
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
	u32 num_node, num_tet, num_prism, num_hex;
	ASSERT(h5f && "Invalid file");
	ASSERT(group_name && "Invalid group name");

	mesh = (Mesh3D*)CdamMallocHost(sizeof(Mesh3D));
	memset(mesh, 0, sizeof(Mesh3D));

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

	return mesh;
}

void Mesh3DDestroy(Mesh3D* mesh) {
	u32 num_elem = Mesh3DNumTet(mesh) + Mesh3DNumPrism(mesh) + Mesh3DNumHex(mesh);
	Mesh3DDataDestroy(Mesh3DHost(mesh));
	Mesh3DDataDestroy(Mesh3DDevice(mesh));
	// CdamFreeDevice(mesh->epart, sizeof(u32) * num_elem);
	CdamFreeHost(mesh->batch_offset, sizeof(u32) * (mesh->num_batch + 1));
	CdamFreeDevice(mesh->batch_ind, sizeof(u32) * num_elem);
	CdamFreeDevice(mesh->color, sizeof(color_t) * num_elem);
	CdamFreeHost(mesh, sizeof(Mesh3D));
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
// void Mesh3DPartition(Mesh3D* mesh, u32 num_part) {
// 	u32 num_elem = Mesh3DNumTet(mesh) + Mesh3DNumPrism(mesh) + Mesh3DNumHex(mesh);
// 	mesh->num_part = num_part;
// 	mesh->epart = CdamMallocDevice(sizeof(u32) * num_elem);
// 	PartitionMesh3DMETIS(mesh, num_part);
// }

void Mesh3DColor(Mesh3D* mesh) {
	u32 num_elem = Mesh3DNumTet(mesh) + Mesh3DNumPrism(mesh) + Mesh3DNumHex(mesh);
	mesh->color = CdamMallocDevice(sizeof(color_t) * num_elem);
	ColorMeshTet(mesh, MAX_COLOR, mesh->color);
	mesh->num_color = GetMaxColor(mesh->color, Mesh3DNumTet(mesh)) + 1;
}

u32 CountValueColorLegacy(color_t*, u32, color_t);
void FindValueColor(color_t*, u32, color_t, u32*);

void Mesh3DGenerateColorBatch(Mesh3D* mesh) {
	u32 num_tet = Mesh3DNumTet(mesh);
	u32 num_prism = Mesh3DNumPrism(mesh);
	u32 num_hex = Mesh3DNumHex(mesh);
	u32 num_elem = num_tet + num_prism + num_hex;
	color_t* color = (color_t*)CdamMallocDevice(sizeof(color_t) * num_elem);
	mesh->batch_ind = (u32*)CdamMallocDevice(sizeof(u32) * num_elem);

	if(num_tet) {
		
		u32 batch_size = 0;
		u32 max_batch_size = 1000;

		ColorMeshTet(mesh, MAX_COLOR, color);
		u32 num_color = GetMaxColor(color, num_tet) + 1;

		mesh->num_batch = num_color;
		mesh->batch_offset = (u32*)CdamMallocHost(sizeof(u32) * (num_color + 1));
		mesh->batch_offset[0] = 0;

		for(u32 c = 0; c < num_color; c++) {
			batch_size = CountValueColorLegacy(color, num_tet, c);
			mesh->batch_offset[c + 1] = mesh->batch_offset[c] + batch_size;
			if(batch_size > max_batch_size) {
				max_batch_size = batch_size;
			}
		}

		u32* batch = (u32*)CdamMallocDevice(sizeof(u32) * max_batch_size);

		for(u32 c = 0; c < num_color; c++) {
			FindValueColor(color, num_tet, c, batch);
			cudaMemcpy(mesh->batch_ind + mesh->batch_offset[c],
								 batch,
								 sizeof(u32) * (mesh->batch_offset[c + 1] - mesh->batch_offset[c]),
								 cudaMemcpyDeviceToDevice);
		}	

	}

	CdamFreeDevice(color, sizeof(color_t) * num_elem);
}

