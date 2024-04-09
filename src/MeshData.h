#ifndef __MESHDATA_H__
#define __MESHDATA_H__

#include "common.h"

__BEGIN_DECLS__

typedef struct H5FileInfo H5FileInfo;
typedef struct Mesh3DData Mesh3DData;
struct Mesh3DData {
	b32 is_host;
	u32 num_node;
	u32 num_tet;
	u32 num_prism;
	u32 num_hex;

	f64* xg; /* xg[3*num_node] */
	u32* ien; /* ien[4*num_tet+6*num_prism+8*num_hex] */
	// u32* ien_tet; /* ien_tet[4*num_tet] */
	// u32* ien_prism; /* ien_prism[6*num_prism] */
	// u32* ien_hex; /* ien_hex[8*num_hex] */
};

#define Mesh3DDataNumNode(data) ((data)->num_node)
#define Mesh3DDataNumTet(data) ((data)->num_tet)
#define Mesh3DDataNumPrism(data) ((data)->num_prism)
#define Mesh3DDataNumHex(data) ((data)->num_hex)
#define Mesh3DDataCoord(data) ((data)->xg)
#define Mesh3DDataTet(data) ((data)->ien + 0)
#define Mesh3DDataPrism(data) ((data)->ien + 4 * Mesh3DDataNumTet(data))
#define Mesh3DDataHex(data) ((data)->ien + 4 * Mesh3DDataNumTet(data) + 6 * Mesh3DDataNumPrism(data))

Mesh3DData* Mesh3DDataCreateHost(u32 num_node, u32 num_tet, u32 num_prism, u32 num_hex);
Mesh3DData* Mesh3DDataCreateDevice(u32 num_node, u32 num_tet, u32 num_prism, u32 num_hex);
Mesh3DData* Mesh3DDataCreateH5(H5FileInfo* h5f, const char* group_name);
void Mesh3DDataDestroy(Mesh3DData* data);

void Mesh3DDataCopy(Mesh3DData* dst, Mesh3DData* src, MemCopyKind kind);

__END_DECLS__

#endif /* __MESHDATA_H__ */
