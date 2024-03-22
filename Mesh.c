
#include <stdlib.h>
#include <cuda_runtime.h>

#include "Mesh.h"


static void
Mesh3DDataCreateHost(Mesh3DData** data) {
	Mesh3DData* d = (Mesh3DData*)malloc(sizeof(Mesh3DData));
	memset(d, 0, sizeof(Mesh3DData));
	d->is_host = true;
	*data = d;
}

static void
Mesh3DDataCreateHostInit(Mesh3DData* data,
												 u32 num_node,
												 u32 num_tet,
												 u32 num_prism,
												 u32 num_hex) {
	assert(data->is_host);
	data->xg = (f64*)malloc(num_node * 3 * sizeof(f64));
	data->ien_tet = (u32*)malloc(num_tet * 4 * sizeof(u32));
	data->ien_prism = (u32*)malloc(num_prism * 6 * sizeof(u32));
	data->ien_hex = (u32*)malloc(num_hex * 8 * sizeof(u32));
	memset(data->xg, 0, num_node * 3 * sizeof(f64));
	memset(data->ien_tet, 0, num_tet * 4 * sizeof(u32));
	memset(data->ien_prism, 0, num_prism * 6 * sizeof(u32));
	memset(data->ien_hex, 0, num_hex * 8 * sizeof(u32));
}

static void
Mesh3DDataDestroyHost(Mesh3DData* data) {
	free(data->xg);
	free(data->ien_tet);
	free(data->ien_prism);
	free(data->ien_hex);
	free(data);
}

static void
Mesh3DDataCreateDevice(Mesh3DData** data) {
	Mesh3DDataCreateHost(data);
	data->is_host = false;
}

static void
Mesh3DDataCreateDeviceInit(Mesh3DData* data,
													 u32 num_node,
													 u32 num_tet,
													 u32 num_prism,
													 u32 num_hex) {
	assert(!data->is_host);
	cudaMalloc((void**)&data->xg, num_node * 3 * sizeof(f64));
	cudaMalloc((void**)&data->ien_tet, num_tet * 4 * sizeof(u32));
	cudaMalloc((void**)&data->ien_prism, num_prism * 6 * sizeof(u32));
	cudaMalloc((void**)&data->ien_hex, num_hex * 8 * sizeof(u32));
	cudaMemset(data->xg, 0, num_node * 3 * sizeof(f64));
	cudaMemset(data->ien_tet, 0, num_tet * 4 * sizeof(u32));
	cudaMemset(data->ien_prism, 0, num_prism * 6 * sizeof(u32));
	cudaMemset(data->ien_hex, 0, num_hex * 8 * sizeof(u32));
}

static void
Mesh3DDataDestroyDevice(Mesh3DData* data) {
	cudaFree(data->xg);
	cudaFree(data->ien_tet);
	cudaFree(data->ien_prism);
	cudaFree(data->ien_hex);
	free(data);
}

static void
Mesh3DDataCopy(Mesh3DData* dst, Mesh3DData* src) {
	assert(!!dst && "dst is not allocated");
	assert(!!src && "src is not allocated");
	MemCopyKind kind;
	if(dst->is_host && src->is_host) {
		kind = H2H;
	} else if (!dst->is_host && src->is_host) {
		kind = H2D;
	} else if (dst->is_host && !src->is_host) {
		kind = D2H;
	} else {
		kind = D2D;
	}

	if (kind == H2H) {
		memcpy(dst->xg, src->xg, src->num_node * 3 * sizeof(f64));
		memcpy(dst->ien_tet, src->ien_tet, src->num_tet * 4 * sizeof(u32));
		memcpy(dst->ien_prism, src->ien_prism, src->num_prism * 6 * sizeof(u32));
		memcpy(dst->ien_hex, src->ien_hex, src->num_hex * 8 * sizeof(u32));
	}
	else {
		cudaMemcpy(dst->xg, src->xg, src->num_node * 3 * sizeof(f64), kind);
		cudaMemcpy(dst->ien_tet, src->ien_tet, src->num_tet * 4 * sizeof(u32), kind);
		cudaMemcpy(dst->ien_prism, src->ien_prism, src->num_prism * 6 * sizeof(u32), kind);
		cudaMemcpy(dst->ien_hex, src->ien_hex, src->num_hex * 8 * sizeof(u32), kind);
	}

}


void Mesh3DCreate(Mesh3D** mesh) {
	Mesh3D* m = (Mesh3D*)malloc(sizeof(Mesh3D));
	memset(m, 0, sizeof(Mesh3D));
	*mesh = m;
}

void Mesh3DLoad(Mesh3D* mesh, const char* filename) {
	Mesh3DDataCreateHost(&mesh->host);
	Mesh3DLoadHost(mesh->host, filename);
}

void Mesh3DMove(Mesh3D* mesh, MemCopyKind kind) {
	if (kind == H2D) {
		if(!mesh->device) {
			Mesh3DDataCreateDevice(&mesh->device);
			Mesh3DDataCreateDeviceInit(mesh->device, mesh->num_node, mesh->num_tet, mesh->num_prism, mesh->num_hex);
		}
		Mesh3DDataCopyHostToDevice(mesh->host, mesh->device);
	} else if (kind == D2H) {
		Mesh3DDataCopyDeviceToHost(mesh->device, mesh->host);
	}
}	

void Mesh3DDestroy(Mesh3D* mesh) {

	Mesh3DDataDestroyHost(mesh->host);
	if (mesh->device) {
		Mesh3DDataDestroyDevice(mesh->device);
	}
	free(mesh);
}
