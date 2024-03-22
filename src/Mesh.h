#ifndef __MESH_H__

#define __MESH_H__

__BEGIN_DECLS__

#include "common.h"

typedef struct Mesh3DData Mesh3DData;
struct Mesh3DData
{
	bool is_host;
	f64* xg; /* xg[3*nNode] */
	u32* ien_tet; /* ien_tet[4*nElem] */
	u32* ien_prism; /* ien_prism[6*nElem] */
	u32* ien_hex; /* ien_hex[8*nElem] */
};


typedef struct Mesh3D Mesh3D;
struct Mesh3D {
	u32 num_node;
	u32 num_tet;
	u32 num_prism;
	u32 num_hex;

	Mesh3DData* host;
	Mesh3DData* device;
};




void Mesh3DCreate(Mesh3D** mesh);
void Mesh3DDestroy(Mesh3D* mesh);

void Mesh3DLoad(Mesh3D* mesh, const char* filename);
void Mesh3DMove(Mesh3D* mesh, MemCopyKind kind);



__END_DECLS__

#endif //__MESH_H__
