#ifndef __COMMON_H__
#define __COMMON_H__

#if defined(__cplusplus)
#define __BEGIN_DECLS__ extern "C" {
#define __END_DECLS__ }
#else
#define __BEGIN_DECLS__ /* empty */
#define __END_DECLS__ /* empty */
#endif


#include <stdlib.h>
#include <stdint.h>

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

#include <stddef.h>
typedef ptrdiff_t size;

#if defined(NDEBUG)
#	define ASSERT(expr) /* empty */
#elif defined(__GNUC__) // GCC, Clang
#	define ASSERT(expr) while (!(expr)) __builtin_unreachable()
#elif defined(_MSC_VER) // MSVC
#	define ASSERT(expr) while (!(expr)) __debugbreak()
#else
#	define ASSERT(expr) while (!(expr)) *(volatile int *)0 = 0
#endif


#define ASSERT_MSG(expr, msg) ASSERT(expr && msg)

#define UNUSED(args) ((void)(args))




#include <cuda_runtime.h>
typedef enum cudaMemcpyKind MemCopyKind;
#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)
#define D2D (cudaMemcpyDeviceToDevice)
#define H2H (cudaMemcpyHostToHost)

#endif // __COMMON_H__
