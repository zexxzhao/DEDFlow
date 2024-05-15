#ifndef __MESH_H__

#define __MESH_H__


#include "common.h"
#include "MeshData.h"
#include "color.h"

__BEGIN_DECLS__
typedef struct H5FileInfo H5FileInfo;

typedef struct Mesh3D Mesh3D;
struct Mesh3D {
	u32 num_node;
	u32 num_tet;
	u32 num_prism;
	u32 num_hex;

	Mesh3DData* host;
	Mesh3DData* device;
	
	/* Partition */
	u32 num_part;
	u32* epart;

	/* color */
	color_t num_color;
	color_t* color;
};

#define Mesh3DHost(mesh) ((mesh)->host)
#define Mesh3DDevice(mesh) ((mesh)->device)
#define Mesh3DNumNode(mesh) ((mesh)->num_node)
#define Mesh3DNumElem(mesh) ((mesh)->num_tet + (mesh)->num_prism + (mesh)->num_hex)
#define Mesh3DNumTet(mesh) ((mesh)->num_tet)
#define Mesh3DNumPrism(mesh) ((mesh)->num_prism)
#define Mesh3DNumHex(mesh) ((mesh)->num_hex)


Mesh3D* Mesh3DCreate(u32 num_node, u32 num_tet, u32 num_prism, u32 num_hex);
Mesh3D* Mesh3DCreateH5(H5FileInfo* h5f, const char* group_name);
void Mesh3DDestroy(Mesh3D* mesh);

void Mesh3DUpdateHost(Mesh3D* mesh);
void Mesh3DUpdateDevice(Mesh3D* mesh);

void Mesh3DPartition(Mesh3D* mesh, u32 num_part);
void Mesh3DColor(Mesh3D* mesh);

__END_DECLS__

#endif /* __MESH_H__ */


