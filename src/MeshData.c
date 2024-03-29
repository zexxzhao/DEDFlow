
#include <string.h>
#include "alloc.h"
#include "h5util.h"
#include "MeshData.h"

Mesh3DData* Mesh3DDataCreateHost(u32 num_node,
																 u32 num_tet,
																 u32 num_prism,
																 u32 num_hex) {
	Mesh3DData* data = (Mesh3DData*)CdamMallocHost(sizeof(Mesh3DData));

	ASSERT(num_node >= 4 && "Invalid number of nodes");
	ASSERT(num_tet + num_prism + num_hex && "Invalid number of elements");
	memset(data, 0, sizeof(Mesh3DData));

	data->is_host = TRUE;
	Mesh3DDataNumNode(data) = num_node;
	Mesh3DDataNumTet(data) = num_tet;
	Mesh3DDataNumPrism(data) = num_prism;
	Mesh3DDataNumHex(data) = num_hex;

	if (num_node) {
		Mesh3DDataCoord(data) = (f64*)CdamMallocHost(sizeof(f64) * num_node * 3);
		memset(Mesh3DDataCoord(data), 0, num_node * 3 * sizeof(f64));
	}
	if (num_tet) {
		Mesh3DDataTet(data) = (u32*)CdamMallocHost(sizeof(u32) * num_tet * 4);
		memset(Mesh3DDataTet(data), 0, num_tet * 4 * sizeof(u32));
	}
	if (num_prism) {
		Mesh3DDataPrism(data) = (u32*)CdamMallocHost(sizeof(u32) * num_prism * 6);
		memset(Mesh3DDataPrism(data), 0, num_prism * 6 * sizeof(u32));
	}
	if (num_hex) {
		Mesh3DDataHex(data) = (u32*)CdamMallocHost(sizeof(u32) * num_hex * 8);
		memset(Mesh3DDataHex(data), 0, num_hex * 8 * sizeof(u32));
	}
	return data;
}

Mesh3DData* Mesh3DDataCreateDevice(u32 num_node, u32 num_tet, u32 num_prism, u32 num_hex) {

	Mesh3DData* data = (Mesh3DData*)CdamMallocHost(sizeof(Mesh3DData));
	ASSERT(num_node >= 4 && "Invalid number of nodes");
	ASSERT(num_tet + num_prism + num_hex && "Invalid number of elements");
	memset(data, 0, sizeof(Mesh3DData));

	data->is_host = FALSE;
	Mesh3DDataNumNode(data) = num_node;
	Mesh3DDataNumTet(data) = num_tet;
	Mesh3DDataNumPrism(data) = num_prism;
	Mesh3DDataNumHex(data) = num_hex;

	if (num_node) {
		Mesh3DDataCoord(data) = (f64*)CdamMallocDevice(sizeof(f64) * num_node * 3);
		cudaMemset(Mesh3DDataCoord(data), 0, num_node * 3 * sizeof(f64));
	}
	if (num_tet) {
		Mesh3DDataTet(data) = (u32*)CdamMallocDevice(sizeof(u32) * num_tet * 4);
		cudaMemset(Mesh3DDataTet(data), 0, num_tet * 4 * sizeof(u32));
	}
	if (num_prism) {
		Mesh3DDataPrism(data) = (u32*)CdamMallocDevice(sizeof(u32) * num_prism * 6);
		cudaMemset(Mesh3DDataPrism(data), 0, num_prism * 6 * sizeof(u32));
	}
	if (num_hex) {
		Mesh3DDataHex(data) = (u32*)CdamMallocDevice(sizeof(u32) * num_hex * 8);
		cudaMemset(Mesh3DDataHex(data), 0, num_hex * 8 * sizeof(u32));
	}
	return data;
}

Mesh3DData* Mesh3DDataCreateH5(H5FileInfo* h5f, const char* group_name) {
	char buff[256];
	Mesh3DData* data;
	u32 num_node, num_tet, num_prism, num_hex;

	ASSERT(H5FileIsReadable(h5f) && "Mesh3DDataCreateH5: Invalid file");
	ASSERT(strlen(group_name) < 192 && "Mesh3DDateCreateH5: Invalid group name.");

	sprintf(buff, "%s/xg", group_name);
	H5GetDatasetSize(h5f, buff, &num_node);
	ASSERT(num_node % 3 == 0 && "Invalid number of nodes");
	num_node /= 3;	

	sprintf(buff, "%s/ien/tet", group_name);
	H5GetDatasetSize(h5f, buff, &num_tet);
	ASSERT(num_tet % 4 == 0 && "Invalid number of tetrahedra");
	num_tet /= 4;

	sprintf(buff, "%s/ien/prism", group_name);
	H5GetDatasetSize(h5f, buff, &num_prism);
	ASSERT(num_prism % 6 == 0 && "Invalid number of prisms");
	num_prism /= 6;

	sprintf(buff, "%s/ien/hex", group_name);
	H5GetDatasetSize(h5f, buff, &num_hex);
	ASSERT(num_hex % 8 == 0 && "Invalid number of hexahedra");
	num_hex /= 8;

	data = Mesh3DDataCreateHost(num_node, num_tet, num_prism, num_hex);

	if (num_node) {
		sprintf(buff, "%s/xg", group_name);
		H5ReadDatasetf64(h5f, buff, Mesh3DDataCoord(data));
	}

	if (num_tet) {
		sprintf(buff, "%s/ien/tet", group_name);
		H5ReadDatasetu32(h5f, buff, Mesh3DDataTet(data));
	}

	if (num_prism) {
		sprintf(buff, "%s/ien/prism", group_name);
		H5ReadDatasetu32(h5f, buff, Mesh3DDataPrism(data));
	}

	if (num_hex) {
		sprintf(buff, "%s/ien/hex", group_name);
		H5ReadDatasetu32(h5f, buff, Mesh3DDataHex(data));
	}

	return data;

}

void Mesh3DDataDestroy(Mesh3DData* data) {
	if (data->is_host) {
		CdamFreeHost(Mesh3DDataCoord(data), sizeof(f64) * Mesh3DDataNumNode(data) * 3);
		CdamFreeHost(Mesh3DDataTet(data), sizeof(u32) * Mesh3DDataNumTet(data) * 4);
		CdamFreeHost(Mesh3DDataPrism(data), sizeof(u32) * Mesh3DDataNumPrism(data) * 6);
		CdamFreeHost(Mesh3DDataHex(data), sizeof(u32) * Mesh3DDataNumHex(data) * 8);
	}
	else {
		CdamFreeDevice(Mesh3DDataCoord(data), sizeof(f64) * Mesh3DDataNumNode(data) * 3);
		CdamFreeDevice(Mesh3DDataTet(data), sizeof(u32) * Mesh3DDataNumTet(data) * 4);
		CdamFreeDevice(Mesh3DDataPrism(data), sizeof(u32) * Mesh3DDataNumPrism(data) * 6);
		CdamFreeDevice(Mesh3DDataHex(data), sizeof(u32) * Mesh3DDataNumHex(data) * 8);
	}
	Mesh3DDataNumNode(data) = 0;
	Mesh3DDataNumTet(data) = 0;
	Mesh3DDataNumPrism(data) = 0;
	Mesh3DDataNumHex(data) = 0;

	Mesh3DDataCoord(data) = NULL;
	Mesh3DDataTet(data) = NULL;
	Mesh3DDataPrism(data) = NULL;
	Mesh3DDataHex(data) = NULL;

	CdamFreeHost(data, sizeof(Mesh3DData));
}

static b32
CheckMemCompatibility(Mesh3DData* dst, Mesh3DData* src, MemCopyKind kind) {
	if (kind == H2H) {
		return dst->is_host && src->is_host;
	}
	else if (kind == D2D) {
		return !dst->is_host && !src->is_host;
	}
	else if (kind == H2D) {
		return !dst->is_host && src->is_host;
	}
	else if (kind == D2H) {
		return dst->is_host && !src->is_host;
	}
	else {
		ASSERT(FALSE && "Invalid copy kind");
		return FALSE;
	}
}

void Mesh3DDataCopy(Mesh3DData* dst, Mesh3DData* src, MemCopyKind kind) {

	f64 *dst_xg, *src_xg;
	u32 *dst_tet, *src_tet;
	u32 *dst_prism, *src_prism;
	u32 *dst_hex, *src_hex;

	u32 len_buff_node, len_buff_tet, len_buff_prism, len_buff_hex;

	ASSERT(dst && src && "Invalid data");
	if (src == dst) {
		return;
	}

	ASSERT(CheckMemCompatibility(dst, src, kind) && "Invalid memory compatibility");

	ASSERT(Mesh3DDataNumNode(dst) == Mesh3DDataNumNode(src) && "Invalid number of nodes");
	ASSERT(Mesh3DDataNumTet(dst) == Mesh3DDataNumTet(src) && "Invalid number of tetrahedra");
	ASSERT(Mesh3DDataNumPrism(dst) == Mesh3DDataNumPrism(src) && "Invalid number of prisms");
	ASSERT(Mesh3DDataNumHex(dst) == Mesh3DDataNumHex(src) && "Invalid number of hexahedra");

	dst_xg = Mesh3DDataCoord(dst);
	dst_tet = Mesh3DDataTet(dst);
	dst_prism = Mesh3DDataPrism(dst);
	dst_hex = Mesh3DDataHex(dst);

	src_xg = Mesh3DDataCoord(src);
	src_tet = Mesh3DDataTet(src);
	src_prism = Mesh3DDataPrism(src);
	src_hex = Mesh3DDataHex(src);

	len_buff_node = Mesh3DDataNumNode(src) * 3 * sizeof(f64);
	len_buff_tet = Mesh3DDataNumTet(src) * 4 * sizeof(u32);
	len_buff_prism = Mesh3DDataNumPrism(src) * 6 * sizeof(u32);
	len_buff_hex = Mesh3DDataNumHex(src) * 8 * sizeof(u32);

	ASSERT(!!dst_xg == !!len_buff_node && "Invalid buffer");
	ASSERT(!!dst_tet == !!len_buff_tet && "Invalid buffer");
	ASSERT(!!dst_prism == !!len_buff_prism && "Invalid buffer");
	ASSERT(!!dst_hex == !!len_buff_hex && "Invalid buffer");

	if (len_buff_node) {
		cudaMemcpy(dst_xg, src_xg, len_buff_node, kind);
	}

	if (len_buff_tet) {
		cudaMemcpy(dst_tet, src_tet, len_buff_tet, kind);
	}

	if (len_buff_prism) {
		cudaMemcpy(dst_prism, src_prism, len_buff_prism, kind);
	}

	if (len_buff_hex) {
		cudaMemcpy(dst_hex, src_hex, len_buff_hex, kind);
	}
}


