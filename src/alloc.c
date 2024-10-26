
#include <string.h>
#include "alloc.h"

#ifdef CDAM_USE_CUDA
#include <cuda_runtime.h>
#endif

__BEGIN_DECLS__


static void* DefaultMallocHostPrivate(ptrdiff_t size, void* ctx) {
	void* p;
	UNUSED(ctx);
	p = malloc(size);
	ASSERT(p && "Out of memory");
	return p;
}

static void DefaultFreeHostPrivate(void* p, ptrdiff_t size, void* ctx) {
	UNUSED(size);
	UNUSED(ctx);
	free(p);
}

static void DefaultMemsetHostPrivate(void* p, int c, ptrdiff_t size, void* ctx) {
	UNUSED(ctx);
	memset(p, c, size);
}

#ifdef CDAM_USE_CUDA
static void* DefaultMallocDevicePrivate(ptrdiff_t size, void* ctx) {
	void* p;
	UNUSED(ctx);
	CUGUARD(cudaMalloc(&p, size));
	ASSERT(p && "Out of memory");
	return p;
}

static void DefaultFreeDevicePrivate(void* p, ptrdiff_t size, void* ctx) {
	UNUSED(size);
	UNUSED(ctx);
	CUGUARD(cudaFree(p));
}

#endif

static Allocator _default_allocator[] = {
	{NULL, NULL, NULL}, /* Dummy */
	{DefaultMallocHostPrivate, DefaultFreeHostPrivate, NULL},
#ifdef CDAM_USE_CUDA
	{DefaultMallocDevicePrivate, DefaultFreeDevicePrivate, NULL}
#endif
};


Allocator* GetDefaultAllocator(int device) {
	ASSERT(device == HOST_MEM || device == DEVICE_MEM);
	return _default_allocator + device;
}

void* CdamMemset(void* ptr, int value, size_t count, MemType type) {
#ifdef CDAM_USE_CUDA
	if (type == DEVICE_MEM) {
		CUGUARD(cudaMemset(ptr, value, count));
		return ptr;
	}
#endif
	return memset(ptr, value, count);
}

void* CdamMemcpy(void* dst, const void* src, size_t count, MemType dst_type, MemType src_type) {
#ifdef CDAM_USE_CUDA
	if (dst_type == DEVICE_MEM && src_type == DEVICE_MEM) {
		CUGUARD(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
		return dst;
	}
	if (dst_type == DEVICE_MEM && src_type == HOST_MEM) {
		CUGUARD(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
		return dst;
	}
	if (dst_type == HOST_MEM && src_type == DEVICE_MEM) {
		CUGUARD(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
		return dst;
	}
#endif
	return memcpy(dst, src, count);
}
#ifdef CDAM_USE_CUDA
#define OTHER_LOCATION(location) (1 - location)
void CdamPrefetch(void** ptr, size_t count, MemType dst_location) {
	void* p = CdamTMalloc(char, count, dst_location);
	CdamMemcpy(p, *ptr, count, dst_location, OTHER_LOCATION(dst_location));
	CdamFree(*ptr, count, OTHER_LOCATION(dst_location));
	*ptr = p;
}
#undef OTHER_LOCATION
#else
void CdamPrefetch(void** ptr, size_t count, MemType dst_location) {
	UNUSED(ptr);
	UNUSED(count);
	UNUSED(dst_location);
}
#endif

void ArenaCreate(size_t h_size, size_t d_size, Arena** arena) {
	*arena = CdamTMalloc(Arena, 1, HOST_MEM);
	(*arena)->h_beg = CdamTMalloc(byte, h_size, HOST_MEM);
	(*arena)->h_end = (*arena)->h_beg + h_size;
	(*arena)->d_beg = CdamTMalloc(byte, d_size, DEVICE_MEM);
	(*arena)->d_end = (*arena)->d_beg + d_size;
}

void ArenaDestroy(Arena* arena) {
	byte* h_beg = arena->h_beg;
	byte* h_end = arena->h_end;
	CdamFree(h_beg, h_end - h_beg, HOST_MEM);
	byte* d_beg = arena->d_beg;
	byte* d_end = arena->d_end;
	CdamFree(d_beg, d_end - d_beg, DEVICE_MEM);
	CdamFree(arena, sizeof(Arena), HOST_MEM);
}

void* AllocInArena(size_t elem_size, size_t count, Arena* scratch, int flag) {
	byte** beg, **end, *p;
	ptrdiff_t available;
	MemType mem_type;

	if(flag & ARENA_ON_HOST) {
		beg = &scratch->h_beg;
		end = &scratch->h_end;
		mem_type = HOST_MEM;
	} else {
		beg = &scratch->d_beg;
		end = &scratch->d_end;
		mem_type = DEVICE_MEM;
	}
	available = *end - *beg;

	if(available < 0 || count > available / elem_size) {
		if(flag & ARENA_SOFTFAIL) {
			return NULL;
		}
		ABORT("Out of memory");
	}

	p = *beg;
	*beg = p + elem_size * count;

	return flag & ARENA_NONZERO ? p : CdamMemset(p, 0, elem_size * count, mem_type);
}


__END_DECLS__
