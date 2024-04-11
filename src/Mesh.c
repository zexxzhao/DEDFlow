
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

/* Include metis for partitioning */
#include "metis.h"

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
	Mesh3DDataDestroy(Mesh3DHost(mesh));
	Mesh3DDataDestroy(Mesh3DDevice(mesh));
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

#include "partition.h"
void Mesh3DPartition(Mesh3D* mesh, u32 num_part) {

	PartitionMesh3DMETIS(mesh, num_part);

}

