
#include <string.h>
#include "alloc.h"

__BEGIN_DECLS__

static void* DefaultMallocHostPrivate(ptrdiff_t size, void* ctx) {
	void* p;
	UNUSED(ctx);
	p = malloc(size);
	memset(p, 0, size);
	ASSERT(p && "Out of memory");
	return p;
}

static void DefaultFreeHostPrivate(void* p, ptrdiff_t size, void* ctx) {
	UNUSED(size);
	UNUSED(ctx);
	free(p);
	p = NULL;
}

static void* DefaultMallocDevicePrivate(ptrdiff_t size, void* ctx) {
	void* p;
	UNUSED(ctx);
	CUGUARD(cudaMalloc(&p, size));
	CUGUARD(cudaMemset(p, 0, size));
	ASSERT(p && "Out of memory");
	return p;
}

static void DefaultFreeDevicePrivate(void* p, ptrdiff_t size, void* ctx) {
	UNUSED(size);
	UNUSED(ctx);
	CUGUARD(cudaFree(p));
}

static Allocator _default_allocator[2] = {
	{DefaultMallocHostPrivate, DefaultFreeHostPrivate, NULL},
	{DefaultMallocDevicePrivate, DefaultFreeDevicePrivate, NULL}
};


Allocator* GetDefaultAllocator(int device) {
	ASSERT(device == HOST || device == DEVICE);
	return _default_allocator + device;
}


__END_DECLS__
