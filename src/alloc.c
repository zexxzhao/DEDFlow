
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

void ArenaCreate(size_t size, MemType type, Arena** arena) {
	*arena = CdamTMalloc(Arena, 1, HOST_MEM);
	(*arena)->mem_type = type;
	(*arena)->beg = CdamTMalloc(byte, size, type);
	(*arena)->end = (*arena)->beg + size;
	(*arena)->ctx = NULL;
}

void ArenaDestroy(Arena* arena) {
	byte* beg = arena->beg;
	byte* end = arena->end;
	CdamFree(beg, end - beg, arena->mem_type);
	CdamFree(arena, sizeof(Arena), HOST_MEM);
}

void* ArenaPush(size_t elem_size, size_t count, void* ctx, int flag) {
	Arena* arena = (Arena*)ctx;
	ptrdiff_t available = arena->end - arena->beg;

	if(available < 0 || count > available / elem_size) {
		if(flag & ARENA_FLAG_SOFTFAIL) {
			return NULL;
		}
		ABORT("Out of memory");
	}

	byte* p = arena->beg;
	arena->beg += elem_size * count;
	return flag & ARENA_FLAG_NONZERO ? p : CdamMemset(p, 0, elem_size * count, arena->mem_type);
}

void ArenaPop(size_t elem_size, size_t count, void* ctx) {
	Arena* arena = (Arena*)ctx;
	byte* p = (byte*)arena->beg;
	arena->beg = p - elem_size * count;
}


__END_DECLS__
