#ifndef __MESH_H__

#define __MESH_H__

__BEGIN_DECLS__

#include "common.h"

typedef struct Mesh3Df64 Mesh3Df64;
struct Mesh3D {
	u32 nNode;
	u32 nElem;
	f64* xg; /* xg[3*nNode] */
	f64* dg; /* dg[3*nElem] */
	u32* offset; /* offset[nElem+1] */
	u32* ien; /* ien[4*nElem] */
};



void Mesh3DHostCreate(Mesh3DHost);
void Mesh3DHostDestory(Mesh3DHost* mesh);

typedef Mesh3D Mesh3DHost;
typedef Mesh3D Mesh3DDevice;



__END_DECLS__

#endif //__MESH_H__
