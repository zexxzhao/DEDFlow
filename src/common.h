#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#include <cuda_runtime.h>

#if defined(__cplusplus) || defined(__CUDACC__)
#define __BEGIN_DECLS__ extern "C" {
#define __END_DECLS__ }
#else
#define __BEGIN_DECLS__ /* empty */
#define __END_DECLS__ /* empty */
#endif

__BEGIN_DECLS__
/* Signed integer */
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

/* Unsigned integer */
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

/* Floating point */
typedef float f32;
typedef double f64;

typedef char byte;
typedef ptrdiff_t size;

#ifdef USE_F64_VALUE
typedef f64 value_type;
#elif defined(USE_F32_VALUE)
typedef f32 value_type;
#else
#warning "No value type specified, using f64"
typedef f64 value_type;
#endif

#if defined(USE_U32_INDEX)
typedef u32 index_type;
#elif defined(USE_U64_INDEX)
typedef u64 index_type;
#elif defined(USE_I32_INDEX)
typedef i32 index_type;
#elif defined(USE_I64_INDEX)
typedef i64 index_type;
#else
#warning "No index type specified, using i32"
typedef i32 index_type;
#endif

/* Boolean */
typedef int32_t b32;
#define TRUE (1==1)
#define FALSE (1==0)
#define BOOL_C(x) (!!(x))

#define SIZE_OF(x) ((index_type)sizeof(x))

#if defined(NDEBUG)
#	define ASSERT(expr) /* empty */
#elif defined(__GNUC__) // GCC, Clang
#	define ASSERT(expr) while (!(expr)) __builtin_trap()
#elif defined(_MSC_VER) // MSVC
#	define ASSERT(expr) while (!(expr)) __debugbreak()
#else
#	define ASSERT(expr) while (!(expr)) *(volatile int *)0 = 0
#endif


#define UNUSED(args) ((void)(args))



typedef enum cudaMemcpyKind MemCopyKind;
#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)
#define D2D (cudaMemcpyDeviceToDevice)
#define H2H (cudaMemcpyHostToHost)

#define CUGUARD(err) do { GPUAssertPrivate((err), __FILE__, __LINE__); } while (0)

static inline void __host__
GPUAssertPrivate(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		printf("GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
		ASSERT(FALSE);
	}
}

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__END_DECLS__
#endif /* __COMMON_H__ */
