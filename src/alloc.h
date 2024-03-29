#ifndef __ALLOC_H__
#define __ALLOC_H__

#include "common.h"

__BEGIN_DECLS__

typedef enum DeviceType DeviceType;
enum DeviceType {
	HOST = 0,
	DEVICE = 1,
};


#define MEM_LEN_DEFAULT (-1);

typedef void* UserCtxPtr;

typedef struct Allocator Allocator;
struct Allocator {
	void* (*malloc)(ptrdiff_t, UserCtxPtr);
	void (*free)(void*, ptrdiff_t, UserCtxPtr);
	UserCtxPtr ctx;
};

Allocator* GetDefaultAllocator(int device_id);

#define CdamMallocHost(count) \
		(GetDefaultAllocator(HOST)->malloc((ptrdiff_t)(count), GetDefaultAllocator(HOST)->ctx))
#define CdamFreeHost(ptr, count) \
		(GetDefaultAllocator(HOST)->free(ptr, (ptrdiff_t)(count), GetDefaultAllocator(HOST)->ctx))

#define CdamMallocDevice(count) \
		(GetDefaultAllocator(DEVICE)->malloc((ptrdiff_t)(count), GetDefaultAllocator(DEVICE)->ctx))
#define CdamFreeDevice(ptr, count) \
		(GetDefaultAllocator(DEVICE)->free(ptr, (ptrdiff_t)(count), GetDefaultAllocator(DEVICE)->ctx))

__END_DECLS__

#endif /* __ALLOC_H__ */
