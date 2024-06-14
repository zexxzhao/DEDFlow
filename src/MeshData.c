
#include <string.h>
#include "alloc.h"
#include "h5util.h"
#include "MeshData.h"

Mesh3DData* Mesh3DDataCreateHost(index_type num_node,
																 index_type num_tet,
																 index_type num_prism,
																 index_type num_hex) {
	Mesh3DData* data = (Mesh3DData*)CdamMallocHost(SIZE_OF(Mesh3DData));
	index_type elem_buff_size = num_tet * 4 + num_prism * 6 + num_hex * 8;

	ASSERT(num_node >= 4 && "Invalid number of nodes");
	ASSERT(num_tet + num_prism + num_hex && "Invalid number of elements");
	memset(data, 0, SIZE_OF(Mesh3DData));

	data->is_host = TRUE;
	Mesh3DDataNumNode(data) = num_node;
	Mesh3DDataNumTet(data) = num_tet;
	Mesh3DDataNumPrism(data) = num_prism;
	Mesh3DDataNumHex(data) = num_hex;

	if (num_node) {
		Mesh3DDataCoord(data) = (f64*)CdamMallocHost(SIZE_OF(f64) * num_node * 3);
		memset(Mesh3DDataCoord(data), 0, num_node * 3 * SIZE_OF(f64));
	}
	data->ien = (index_type*)CdamMallocHost(SIZE_OF(index_type) * elem_buff_size);
	memset(data->ien, 0, elem_buff_size * SIZE_OF(index_type));
	return data;
}

Mesh3DData* Mesh3DDataCreateDevice(index_type num_node, index_type num_tet, index_type num_prism, index_type num_hex) {

	Mesh3DData* data = (Mesh3DData*)CdamMallocHost(SIZE_OF(Mesh3DData));
	index_type elem_buff_size = num_tet * 4 + num_prism * 6 + num_hex * 8;

	ASSERT(num_node >= 4 && "Invalid number of nodes");
	ASSERT(num_tet + num_prism + num_hex && "Invalid number of elements");
	memset(data, 0, SIZE_OF(Mesh3DData));

	data->is_host = FALSE;
	Mesh3DDataNumNode(data) = num_node;
	Mesh3DDataNumTet(data) = num_tet;
	Mesh3DDataNumPrism(data) = num_prism;
	Mesh3DDataNumHex(data) = num_hex;

	if (num_node) {
		Mesh3DDataCoord(data) = (f64*)CdamMallocDevice(SIZE_OF(f64) * num_node * 3);
		CUGUARD(cudaMemset(Mesh3DDataCoord(data), 0, num_node * 3 * SIZE_OF(f64)));
	}
	data->ien = (index_type*)CdamMallocDevice(SIZE_OF(index_type) * elem_buff_size);
	CUGUARD(cudaMemset(data->ien, 0, elem_buff_size * SIZE_OF(index_type)));
	return data;
}

Mesh3DData* Mesh3DDataCreateH5(H5FileInfo* h5f, const char* group_name) {
	char buff[256];
	Mesh3DData* data;
	index_type num_node, num_tet, num_prism, num_hex;

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
		H5ReadDatasetInd(h5f, buff, Mesh3DDataTet(data));
	}

	if (num_prism) {
		sprintf(buff, "%s/ien/prism", group_name);
		H5ReadDatasetInd(h5f, buff, Mesh3DDataPrism(data));
	}

	if (num_hex) {
		sprintf(buff, "%s/ien/hex", group_name);
		H5ReadDatasetInd(h5f, buff, Mesh3DDataHex(data));
	}

	return data;

}

void Mesh3DDataDestroy(Mesh3DData* data) {
	index_type elem_buff_size = Mesh3DDataNumTet(data) * 4 + Mesh3DDataNumPrism(data) * 6 + Mesh3DDataNumHex(data) * 8;
	if (data->is_host) {
		CdamFreeHost(Mesh3DDataCoord(data), SIZE_OF(f64) * Mesh3DDataNumNode(data) * 3);
		CdamFreeHost(data->ien, SIZE_OF(index_type) * elem_buff_size);
	}
	else {
		CdamFreeDevice(Mesh3DDataCoord(data), SIZE_OF(f64) * Mesh3DDataNumNode(data) * 3);
		CdamFreeDevice(data->ien, SIZE_OF(index_type) * elem_buff_size);
	}
	Mesh3DDataNumNode(data) = 0;
	Mesh3DDataNumTet(data) = 0;
	Mesh3DDataNumPrism(data) = 0;
	Mesh3DDataNumHex(data) = 0;

	Mesh3DDataCoord(data) = NULL;
	data->ien = NULL;

	CdamFreeHost(data, SIZE_OF(Mesh3DData));
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
	index_type *dst_tet, *src_tet;
	index_type *dst_prism, *src_prism;
	index_type *dst_hex, *src_hex;

	index_type len_buff_node, len_buff_tet, len_buff_prism, len_buff_hex;

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

	len_buff_node = Mesh3DDataNumNode(src) * 3 * SIZE_OF(f64);
	len_buff_tet = Mesh3DDataNumTet(src) * 4 * SIZE_OF(index_type);
	len_buff_prism = Mesh3DDataNumPrism(src) * 6 * SIZE_OF(index_type);
	len_buff_hex = Mesh3DDataNumHex(src) * 8 * SIZE_OF(index_type);

	ASSERT(!!dst_xg == !!len_buff_node && "Invalid buffer");
	ASSERT(!!dst_tet == !!len_buff_tet && "Invalid buffer");
	ASSERT(!!dst_prism == !!len_buff_prism && "Invalid buffer");
	ASSERT(!!dst_hex == !!len_buff_hex && "Invalid buffer");

	if (len_buff_node) {
		CUGUARD(cudaMemcpy(dst_xg, src_xg, len_buff_node, kind));
	}

	if (len_buff_tet) {
		CUGUARD(cudaMemcpy(dst_tet, src_tet, len_buff_tet, kind));
	}

	if (len_buff_prism) {
		CUGUARD(cudaMemcpy(dst_prism, src_prism, len_buff_prism, kind));
	}

	if (len_buff_hex) {
		CUGUARD(cudaMemcpy(dst_hex, src_hex, len_buff_hex, kind));
	}
}


