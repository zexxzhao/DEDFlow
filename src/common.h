#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#ifdef CDAM_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#endif



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

typedef u8 byte;
typedef ptrdiff_t size;

#define MPI_BYTE_TYPE MPI_BYTE
#ifdef USE_F64_VALUE
typedef f64 value_type;
#define MPI_VALUE_TYPE MPI_DOUBLE
#elif defined(USE_F32_VALUE)
typedef f32 value_type;
#define MPI_VALUE_TYPE MPI_FLOAT
#else
#warning "No value type specified, using f64"
typedef f64 value_type;
#endif

#if defined(USE_U32_INDEX)
typedef u32 index_type;
#define MPI_INDEX_TYPE MPI_UNSIGNED
#elif defined(USE_U64_INDEX)
typedef u64 index_type;
#define MPI_INDEX_TYPE MPI_UNSIGNED_LONG
#elif defined(USE_I32_INDEX)
typedef i32 index_type;
#define MPI_INDEX_TYPE MPI_INT
#elif defined(USE_I64_INDEX)
typedef i64 index_type;
#define MPI_INDEX_TYPE MPI_LONG
#else
#warning "No index type specified, using i32"
typedef i32 index_type;
#define MPI_INDEX_TYPE MPI_INT
#endif

typedef enum {
	CDAM_I8 = 0,
	CDAM_I16 = 1,
	CDAM_I32 = 2,
	CDAM_I64 = 3,
	CDAM_U8 = 4,
	CDAM_U16 = 5,
	CDAM_U32 = 6,
	CDAM_U64 = 7,
	CDAM_F32 = 8,
	CDAM_F64 = 9,
	CDAM_BYTE = 10,
} CdamDataType;

#define BYTIFY(x) ((byte*)(x))

/* Boolean */
typedef int32_t b32;
#define TRUE (1==1)
#define FALSE (1==0)
#define BOOL_C(x) (!!(x))

#define SIZE_OF(x) ((index_type)sizeof(x))
/* Redefine sizeof to index_type
 * This is to avoid implicit conversion from size_t to index_type
 * Use ellipsis to be compatible with c++ 
 */
#define sizeof(...) ((index_type)sizeof(__VA_ARGS__))

#if defined(NDEBUG)
#	define ASSERT(expr) /* empty */
#elif defined(__GNUC__) // GCC, Clang
#	define ASSERT(expr) while (!(expr)) __builtin_trap()
#elif defined(_MSC_VER) // MSVC
#	define ASSERT(expr) while (!(expr)) __debugbreak()
#else
#	define ASSERT(expr) while (!(expr)) *(volatile int *)0 = 0
#endif

#if defined(__GNUC__) // GCC, Clang
#define ABORT(msg) do { fprintf(stderr, "%s\n", msg); __builtin_trap(); } while (0)
#elif defined(_MSC_VER) // MSVC
#define ABORT(msg) do { fprintf(stderr, "%s\n", msg); __debugbreak(); } while (0)
#else
#define ABORT(msg) do { fprintf(stderr, "%s\n", msg); *(volatile int *)0 = 0; } while (0)
#endif


#define UNUSED(args) ((void)(args))

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define MAX(a, b) ({ \
	__typeof__(a) _a = (a); \
	__typeof__(b) _b = (b); \
	_a > _b ? _a : _b; \
})
#define MIN(a, b) ({ \
	__typeof__(a) _a = (a); \
	__typeof__(b) _b = (b); \
	_a < _b ? _a : _b; \
})



typedef enum cudaMemcpyKind MemCopyKind;
/*
#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)
#define D2D (cudaMemcpyDeviceToDevice)
#define H2H (cudaMemcpyHostToHost)
*/

void Init(int argc, char** argv);
void Finalize();
enum GlobalContextType {
	GLOBAL_CONTEXT_CUSPARSE_HANDLE = 0,
	GLOBAL_CONTEXT_CUBLAS_HANDLE = 1
};
typedef enum GlobalContextType GlobalContextType;

void* GlobalContextGet(GlobalContextType type); 



#ifdef CDAM_USE_CUDA
#define CUGUARD(err) do { GPUAssertPrivate((err), __FILE__, __LINE__); } while (0)
static inline void
GPUAssertPrivate(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		printf("GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
		ASSERT(FALSE);
	}
}
#else
#define CUGUARD(err) /* empty */
#endif

/* Avoid using malloc/free directly */
// #define malloc malloc_is_forbidden
// #define free free_is_forbidden

__END_DECLS__
#endif /* __COMMON_H__ */
