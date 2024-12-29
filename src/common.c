#include <mpi.h>

#ifdef USE_AMGX
#include <amgx_c.h>
#endif

#include "common.h"
#include "alloc.h"

__BEGIN_DECLS__

struct GlobalCtx {
#ifdef CDAM_USE_CUDA
	cusparseHandle_t cusparse_handle;	
	cublasHandle_t cublas_handle;
#endif
};
typedef struct GlobalCtx GlobalCtx;

static void* global_ctx = NULL;

void Init(int argc, char **argv) {
	// MPI_Init(&argc, &argv);
	MPI_Init(NULL, NULL);
	global_ctx = CdamTMalloc(GlobalCtx, 1, HOST_MEM);
	GlobalCtx *ctx = (GlobalCtx *)global_ctx;
#ifdef CDAM_USE_CUDA
	cublasCreate(&ctx->cublas_handle);
	cusparseCreate(&ctx->cusparse_handle);
#ifdef USE_AMGX
	AMGX_initialize();
#endif
#endif
}

void Finalize() {
	MPI_Finalize();
	GlobalCtx *ctx = (GlobalCtx *)global_ctx;
#ifdef CDAM_USE_CUDA
	cublasDestroy(ctx->cublas_handle);
	cusparseDestroy(ctx->cusparse_handle);
#ifdef USE_AMGX
	AMGX_finalize();
#endif
#endif
	CdamFree(global_ctx, sizeof(GlobalCtx), HOST_MEM);
}

void* GlobalContextGet(GlobalContextType type) {
#ifdef CDAM_USE_CUDA
	GlobalCtx *ctx = (GlobalCtx *)global_ctx;
	switch(type) {
		case GLOBAL_CONTEXT_CUSPARSE_HANDLE:
			return &ctx->cusparse_handle;
		case GLOBAL_CONTEXT_CUBLAS_HANDLE:
			return &ctx->cublas_handle;
		default:
			return NULL;
	}
#else
	return NULL;
#endif
}

__END_DECLS__
