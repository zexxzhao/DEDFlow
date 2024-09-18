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

typedef char byte;
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

#define BYTIFY(x) ((byte*)(x))

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

#if defined(CDAM_USE_CUDA)

#ifdef USE_F64_VALUE
#define BLAS_FUNC_NAME(func) cublasD##func
#elif defined(USE_F32_VALUE)
#define BLAS_FUNC_NAME(func) cublasS##func
#else
#warning "No value type specified, using f64"
#define BLAS_FUNC_NAME(func) cublasD##func
#endif

#define BLAS_CALL(func, ...) \
	do { \
		cublasHandle_t handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE); \
		BLAS_FUNC_NAME(func)(handle, __VA_ARGS__); \
	} while (0)

#define BLAS_SET_POINTER_DEVICE \
	do { \
		cublasHandle_t handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE); \
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE); \
	} while (0)
#define BLAS_SET_POINTER_HOST \
	do { \
		cublasHandle_t handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE); \
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST); \
	} while (0)


#define BLAS_N CUBLAS_OP_N
#define BLAS_T CUBLAS_OP_T
#define BLAS_C CUBLAS_OP_C

#define BLAS_UP CUBLAS_FILL_MODE_UPPER
#define BLAS_LO CUBLAS_FILL_MODE_LOWER

#define BLAS_NONUNIT CUBLAS_DIAG_NON_UNIT
#define BLAS_UNIT CUBLAS_DIAG_UNIT

#if defined(USE_F64_VALUE)
#define SPBLAS_FUNC_NAME(func) cusparseD##func
#elif defined(USE_F32_VALUE)
#define SPBLAS_FUNC_NAME(func) cusparseS##func
#else
#warning "No value type specified, using f64"
#define SPBLAS_FUNC_NAME(func) cusparseD##func
#endif

#define SP_CALL(func, ...) \
	do { \
		cusparseHandle_t handle = *(cusparseHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUSPARSE_HANDLE); \
		SPBLAS_FUNC_NAME(func)(handle, __VA_ARGS__); \
	} while (0)


#else
#include <cblas.h>
#include <lapacke.h>

#if defined(USE_F64_VALUE)
#define BLAS_FUNC_NAME(func) cblas_d##func
#elif defined(USE_F32_VALUE)
#define BLAS_FUNC_NAME(func) cblas_s##func
#else
#warning "No value type specified, using f64"
#define BLAS_FUNC_NAME(func) cblas_d##func
#endif

static inline int BlasLevelPrivate(const char* name) {
	static const char* level1_routine_names[] = {
		/* Blas Level-1 */
		"amin", "amax", "asum", 
		"axpy", "copy", "dot", 
		"nrm2", "rot", "rotg", 
		"rotm", "rotmg", "scal", 
	};	
	static const char* level2_routine_names[] = {
		/* Blas Level-2 */
		"gemv", "gbmv", "ger", 
		"sbmv", "spmv", "spr",
		"spr2", "symv", "syr",
		"syr2", "tbmv", "tbsv",
		"tpmv", "tpsv", "trmv",
		"trsv", 
		/* Blas Level-2 Extended */
		"getrsBatched", "getrfBatched", 
		"gemvBatched", "gemvStridedBatched" 
	};
	static const char* level3_routine_names[] = {
		/* Blas Level-3 */
		"gemm", "syrk", "syr2k", 
		"trmm", "trsm", 
		/* Blas Level-3 Extended */
		"gemmEx", "geam"
	};

	for (int i = 0; i < sizeof(level1_routine_names) / sizeof(level1_routine_names[0]); i++) {
		if (strcmp(name, level1_routine_names[i]) == 0) {
			return 1;
		}
	}
	for (int i = 0; i < sizeof(level2_routine_names) / sizeof(level2_routine_names[0]); i++) {
		if (strcmp(name, level2_routine_names[i]) == 0) {
			return 2;
		}
	}
	for (int i = 0; i < sizeof(level3_routine_names) / sizeof(level3_routine_names[0]); i++) {
		if (strcmp(name, level3_routine_names[i]) == 0) {
			return 3;
		}
	}
	return 0;
}


#define BLAS_CALL(func, ...) \
	do { \
		int level = BlasLevelPrivate(#func); \
		if(level == 1) { \
			BLAS_FUNC_NAME(func)(__VA_ARGS__); \
		} else if(level == 2 || level == 3) { \
			BLAS_FUNC_NAME(func)(CblasColMajor, __VA_ARGS__); \
		} else { \
			fprintf(stderr, "Unknown Blas Level: %s\n", #func); \
			ASSERT(FALSE); \
		} \
	} while (0)

#define BLAS_SET_POINTER_DEVICE /* empty */
#define BLAS_SET_POINTER_HOST /* empty */

#define BLAS_N CblasNoTrans
#define BLAS_T CblasTrans
#define BLAS_C CblasConjTrans

#define BLAS_UP CblasUpper
#define BLAS_LO CblasLower

#define BLAS_NONUNIT CblasNonUnit
#define BLAS_UNIT CblasUnit
#endif


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



__END_DECLS__
#endif /* __COMMON_H__ */
