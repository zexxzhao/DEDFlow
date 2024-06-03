#ifndef __MESH_H__

#define __MESH_H__


#include "common.h"
#include "MeshData.h"
#include "color.h"

__BEGIN_DECLS__
typedef struct H5FileInfo H5FileInfo;

typedef struct Mesh3D Mesh3D;
struct Mesh3D {
	index_type num_node;
	index_type num_tet;
	index_type num_prism;
	index_type num_hex;

	Mesh3DData* host;
	Mesh3DData* device;
	
	/* Partition */
	// index_type num_part;
	// index_type* epart;

	/* Bound */
	index_type num_bound;
	index_type* bound_node_offset;
	index_type* bound_node;
	index_type* bound_elem_offset;
	index_type* bound_ien;
	index_type* bound_f2e;
	index_type* bound_forn;

	/* Batch */
	index_type num_batch;
	index_type* batch_offset;
	index_type* batch_ind;

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

#define Mesh3DBoundNumNode(mesh, i) ((mesh)->bound_node_offset[(i)+1] - (mesh)->bound_node_offset[(i)])
#define Mesh3DBoundNode(mesh, i) ((mesh)->bound_node + (mesh)->bound_node_offset[(i)])

#define Mesh3DBoundNumElem(mesh, i) ((mesh)->bound_elem_offset[(i)+1] - (mesh)->bound_elem_offset[(i)])
#define Mesh3DBoundIEN(mesh, i) ((mesh)->bound_ien + (mesh)->bound_elem_offset[(i)] * 3)
#define Mesh3DBoundF2E(mesh, i) ((mesh)->bound_f2e + (mesh)->bound_elem_offset[(i)])
#define Mesh3DBoundFORN(mesh, i) ((mesh)->bound_forn + (mesh)->bound_elem_offset[(i)])


Mesh3D* Mesh3DCreate(index_type num_node, index_type num_tet, index_type num_prism, index_type num_hex);
Mesh3D* Mesh3DCreateH5(H5FileInfo* h5f, const char* group_name);
void Mesh3DDestroy(Mesh3D* mesh);

void Mesh3DUpdateHost(Mesh3D* mesh);
void Mesh3DUpdateDevice(Mesh3D* mesh);

// void Mesh3DPartition(Mesh3D* mesh, index_type num_part);
void Mesh3DColor(Mesh3D* mesh);
void Mesh3DGenerateColorBatch(Mesh3D* mesh);


__END_DECLS__

#endif /* __MESH_H__ */


