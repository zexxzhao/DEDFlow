#include <cublas_v2.h>
#include <cusparse.h>

#ifdef USE_AMGX
#include <amgx_c.h>
#endif

#include "common.h"
#include "alloc.h"

__BEGIN_DECLS__

struct GlobalCtx {
	cusparseHandle_t cusparse_handle;	
	cublasHandle_t cublas_handle;
};
typedef struct GlobalCtx GlobalCtx;

static void* global_ctx = NULL;

void Init(int argc, char **argv) {
	global_ctx = CdamMallocHost(SIZE_OF(GlobalCtx));
	GlobalCtx *ctx = (GlobalCtx *)global_ctx;
	cusparseCreate(&ctx->cusparse_handle);
	cublasCreate(&ctx->cublas_handle);
#ifdef USE_AMGX
	AMGX_initialize();
#endif
}

void Finalize() {
	GlobalCtx *ctx = (GlobalCtx *)global_ctx;
	cusparseDestroy(ctx->cusparse_handle);
	cublasDestroy(ctx->cublas_handle);
#ifdef USE_AMGX
	AMGX_finalize();
#endif
	CdamFreeHost(global_ctx, SIZE_OF(GlobalCtx));
}

void* GlobalContextGet(GlobalContextType type) {
	GlobalCtx *ctx = (GlobalCtx *)global_ctx;
	switch(type) {
		case GLOBAL_CONTEXT_CUSPARSE_HANDLE:
			return &ctx->cusparse_handle;
		case GLOBAL_CONTEXT_CUBLAS_HANDLE:
			return &ctx->cublas_handle;
		default:
			return NULL;
	}
}

__END_DECLS__
