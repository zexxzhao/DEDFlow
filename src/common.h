#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#include <cuda_runtime.h>

#if defined(__cplusplus)
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

/* Boolean */
typedef int32_t b32;
#define TRUE (1==1)
#define FALSE (1==0)
#define BOOL_C(x) (!!(x))

typedef ptrdiff_t size;

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

static inline void __host__ __device__
GPUAssertPrivate(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
		ASSERT(FALSE);
	}
}
__END_DECLS__
#endif /* __COMMON_H__ */
