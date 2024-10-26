#ifndef __ALLOC_H__
#define __ALLOC_H__

#include "common.h"

__BEGIN_DECLS__

enum DeviceType {
	HOST = 0,
	DEVICE = 1,
};
typedef enum DeviceType DeviceType;


#define MEM_LEN_DEFAULT (-1)


enum MemType {
	HOST_MEM = 1,
#ifdef CDAM_USE_CUDA
	DEVICE_MEM = 2,
	PINNED_MEM = 3,
	UNIFIED_MEM = 4,
#else
	DEVICE_MEM = HOST_MEM,
	PINNED_MEM = HOST_MEM,
	UNIFIED_MEM = HOST_MEM,
#endif
};
typedef enum MemType MemType;


typedef struct Allocator Allocator;
struct Allocator {
	void* (*malloc)(ptrdiff_t, void*);
	void (*free)(void*, ptrdiff_t, void*);
	void* ctx;
};

Allocator* GetDefaultAllocator(int device_id);

#define CdamMallocHost(count) \
		(GetDefaultAllocator(HOST)->malloc((ptrdiff_t)(count), GetDefaultAllocator(HOST)->ctx))
#define CdamFreeHost(ptr, count) \
		(GetDefaultAllocator(HOST)->free(ptr, (ptrdiff_t)(count), GetDefaultAllocator(HOST)->ctx))

#ifdef CDAM_USE_CUDA
#define CdamMallocDevice(count) \
		(GetDefaultAllocator(DEVICE)->malloc((ptrdiff_t)(count), GetDefaultAllocator(DEVICE)->ctx))
#define CdamFreeDevice(ptr, count) \
		(GetDefaultAllocator(DEVICE)->free(ptr, (ptrdiff_t)(count), GetDefaultAllocator(DEVICE)->ctx))
#endif

#define CdamTMalloc(type, count, mem_type) ((type*)CdamMalloc(sizeof(type) * (count), (mem_type)))

static void* CdamMalloc(ptrdiff_t size, MemType mem_type) {
	return GetDefaultAllocator((int)(mem_type))->malloc((ptrdiff_t)(size), GetDefaultAllocator((int)(mem_type))->ctx);
}

static void CdamFree(void* ptr, ptrdiff_t size, MemType mem_type) {
	UNUSED(size);
	GetDefaultAllocator((int)(mem_type))->free(ptr, MEM_LEN_DEFAULT, GetDefaultAllocator((int)(mem_type))->ctx);
	ptr = NULL;
}
void* CdamMemset(void* ptr, int value, size_t count, MemType mem_type);
void* CdamMemcpy(void* dst, const void* src, size_t count, MemType dst_mem_type, MemType src_mem_type);


void CdamPrefetch(void** ptr, size_t count, MemType dst_location);

typedef struct Arena Arena;
struct Arena {
	byte* h_beg;
	byte* h_end;
	byte* d_beg;
	byte* d_end;
	void* ctx;
};

void ArenaCreate(size_t h_size, size_t d_size, Arena** arena); 
void ArenaDestroy(Arena* arena);

#define ARENA_DEFAULT (0) /* Default is on device and zeroed */
#define ARENA_NONZERO (1 << 0)
#define ARENA_ON_HOST (1 << 1)
#define ARENA_SOFTFAIL (1 << 2)
void* AllocInArena(size_t elem_size, size_t count, Arena* arena, int flag);
// void ArenaPop(size_t elem_size, size_t count, void* arena);

__END_DECLS__

#endif /* __ALLOC_H__ */
