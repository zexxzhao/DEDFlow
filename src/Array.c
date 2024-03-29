
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "alloc.h"
#include "h5util.h"
#include "Array.h"

__BEGIN_DECLS__

Array* ArrayCreateHost(u32 len) {
	Array *a;
	ASSERT(len > 0 && "ArrayCreateHost: Invalid length");
	a = (Array*)CdamMallocHost(sizeof(Array));
	memset(a, 0, sizeof(Array));

	a->is_host = TRUE;
	ArrayLen(a) = len;
	ArrayData(a) = (f64*)CdamMallocHost(sizeof(f64) * len);

	return a;
}

Array* ArrayCreateDevice(u32 len) {
	Array *a;
	ASSERT(len > 0 && "ArrayCreateDevice: Invalid length");
	a = (Array*)CdamMallocHost(sizeof(Array));
	memset(a, 0, sizeof(Array));

	a->is_host = FALSE;
	ArrayLen(a) = len;
	ArrayData(a) = (f64*)CdamMallocDevice(sizeof(f64) * len);

	return a;
}

void ArrayDestroy(Array* a) {
	ASSERT(a && "ArrayDestroy: NULL pointer");
	if (a) {
		if (a->is_host) {
			CdamFreeHost(ArrayData(a), ArrayLen(a) * sizeof(f64));
			ArrayData(a) = NULL;
		}
		else {
			CdamFreeDevice(ArrayData(a), ArrayLen(a) * sizeof(f64));
			ArrayData(a) = NULL;
		}
	}
	CdamFreeHost(a, sizeof(Array));
	a = NULL;
}

void ArrayCopy(Array* dst, const Array* src, MemCopyKind kind) {
	ASSERT(dst && src && "ArrayCopy: NULL pointer");
	ASSERT(dst->data && "ArrayCopy: Invalid destination array");
	ASSERT(src->data && "ArrayCopy: Invalid source array");
	ASSERT(dst->len == src->len && "ArrayCopy: Array length mismatch");

	if (kind == H2H) {
		ASSERT(dst->is_host && src->is_host && "ArrayCopy: Invalid memory copy kind");
	}
	else if (kind == H2D) {
		ASSERT(!dst->is_host && src->is_host && "ArrayCopy: Invalid memory copy kind");
	}
	else if (kind == D2H) {
		ASSERT(dst->is_host && !src->is_host && "ArrayCopy: Invalid memory copy kind");
	}
	else if (kind == D2D) {
		ASSERT(!dst->is_host && !src->is_host && "ArrayCopy: Invalid memory copy kind");
	}

	if (kind == H2H) {
		memcpy(dst->data, src->data, sizeof(f64) * src->len);
	}
	else {
		CUGUARD(cudaMemcpy(dst->data, src->data, sizeof(f64) * src->len, kind));
	}
}

void ArraySet(Array* a, f64 val) {
	f64* d_data;
	ASSERT(a && "ArraySet: NULL pointer");

	if (a->is_host) {
		for (u32 i = 0; i < ArrayLen(a); i++) {
			ArrayData(a)[i] = val;
		}
	}
	else {
		d_data = (f64*)CdamMallocHost(sizeof(f64) * ArrayLen(a));
		for (u32 i = 0; i < ArrayLen(a); i++) {
			d_data[i] = val;
		}
		CUGUARD(cudaMemcpy(ArrayData(a), d_data, sizeof(f64) * ArrayLen(a), cudaMemcpyHostToDevice));
		free(d_data);
	}
}

void ArrayZero(Array* a) {
	ArrayScale(a, 0.0);
}

void ArraySetAt(Array* a, u32 n, const u32* idx, const f64* val) {
	ASSERT(a && idx && val && "ArraySetAt: NULL pointer");

	if (a->is_host) {
		for(u32 i = 0; i < n; ++i) {
			ArrayData(a)[idx[i]] = val[i];
		}
	}
	else {
		f64* d_data = (f64*)malloc(sizeof(f64) * ArrayLen(a));
		CUGUARD(cudaMemcpy(d_data, ArrayData(a), sizeof(f64) * ArrayLen(a), cudaMemcpyDeviceToHost));
		for(u32 i = 0; i < n; i++) {
			d_data[idx[i]] = val[i];
		}
		CUGUARD(cudaMemcpy(ArrayData(a), d_data, sizeof(f64) * ArrayLen(a), cudaMemcpyHostToDevice));
		free(d_data);
	}
}

void ArrayAt(const Array* a, u32 n, const u32* idx, f64* val) {
	ASSERT(a && idx && val && "ArrayAt: NULL pointer");

	if (a->is_host) {
		for (u32 i = 0; i < n; i++) {
			val[i] = ArrayData(a)[idx[i]];
		}
	}
	else {
		f64* d_data = (f64*)malloc(sizeof(f64) * ArrayLen(a));
		CUGUARD(cudaMemcpy(d_data, ArrayData(a), sizeof(f64) * ArrayLen(a), cudaMemcpyDeviceToHost));
		for (u32 i = 0; i < n; i++) {
			val[i] = d_data[idx[i]];
		}
		free(d_data);
	}
}

void ArrayScale(Array* a, f64 val) {
	ASSERT(a && "ArrayScale: NULL pointer");

	if (a->is_host) {
		for (u32 i = 0; i < ArrayLen(a); i++) {
			ArrayData(a)[i] *= val;
		}
	}
	else {
		cublasStatus_t status; 
		cublasHandle_t handle = NULL;
		cudaStream_t stream = NULL;
		cublasCreate(&handle);
		cudaStreamCreate(&stream);
		cublasSetStream(handle, stream);
		status = cublasDscal(handle, ArrayLen(a), &val, ArrayData(a), 1);
		ASSERT(status == CUBLAS_STATUS_SUCCESS && "ArrayScale: cublasDscal failed");
		cublasDestroy(handle);
		cudaStreamDestroy(stream);
	}
}

void ArrayDot(f64* result, const Array* a, const Array* b) {
	ASSERT(result && a && b && "ArrayDot: NULL pointer");
	ASSERT(ArrayLen(a) == ArrayLen(b) && "ArrayDot: Array length mismatch");
	ASSERT(a->is_host == b->is_host && "ArrayDot: Array type mismatch");

	if (a->is_host) {
		*result = 0.0;
		for (u32 i = 0; i < ArrayLen(a); i++) {
			*result += ArrayData(a)[i] * ArrayData(b)[i];
		}
	}
	else {
		cublasStatus_t status;
		cublasHandle_t handle;
		cublasCreate(&handle);
		status = cublasDdot(handle, ArrayLen(a), ArrayData(a), 1, ArrayData(b), 1, result);
		ASSERT(status == CUBLAS_STATUS_SUCCESS && "ArrayDot: cublasDdot failed");
		cublasDestroy(handle);
	}
}

void ArrayNorm2(f64* result, const Array* a) {
	ASSERT(result && a && "ArrayNorm2: NULL pointer");

	if(a->is_host) {
		*result = 0.0;
		for (u32 i = 0; i < ArrayLen(a); i++) {
			*result += ArrayData(a)[i] * ArrayData(a)[i];
		}
		*result = sqrt(*result);
	}
	else {
		cublasStatus_t status;
		cublasHandle_t handle;
		cublasCreate(&handle);
		status = cublasDnrm2(handle, ArrayLen(a), ArrayData(a), 1, result);
		ASSERT(status == CUBLAS_STATUS_SUCCESS && "ArrayNorm2: cublasDnrm2 failed");
		cublasDestroy(handle);
	}
}

void ArrayAXPY(Array* y, f64 a, const Array* x) {
	ASSERT(y && x && "ArrayAXPY: NULL pointer");
	ASSERT(ArrayLen(y) == ArrayLen(x) && "ArrayAXPY: Array length mismatch");
	ASSERT(y->is_host == x->is_host && "ArrayAXPY: Array type mismatch");

	if (y->is_host) {
		for (u32 i = 0; i < ArrayLen(y); i++) {
			ArrayData(y)[i] += a * ArrayData(x)[i];
		}
	}
	else {
		cublasStatus_t status;
		cublasHandle_t handle;
		cublasCreate(&handle);
		status = cublasDaxpy(handle, ArrayLen(y), &a, ArrayData(x), 1, ArrayData(y), 1);
		ASSERT(status == CUBLAS_STATUS_SUCCESS && "ArrayAXPY: cublasDaxpy failed");
		cublasDestroy(handle);
	}
}

void ArrayAXPBY(Array* y, f64 a, const Array* x, f64 b) {
	ASSERT(y && x && "ArrayAXPBY: NULL pointer");
	ASSERT(ArrayLen(y) == ArrayLen(x) && "ArrayAXPBY: Array length mismatch");
	ASSERT(y->is_host == x->is_host && "ArrayAXPBY: Array type mismatch");

	if (y->is_host) {
		for (u32 i = 0; i < ArrayLen(y); i++) {
			ArrayData(y)[i] = a * ArrayData(x)[i] + b * ArrayData(y)[i];
		}
	}
	else {
		ArrayScale(y, b);
		ArrayAXPY(y, a, x);
	}
}


void ArrayLoad(Array* a, H5FileInfo* h5f, const char* dataset_name) {
	ASSERT(a && h5f && dataset_name && "ArrayLoad: NULL pointer");
	ASSERT(a->is_host && "ArrayLoad: Array must be host type");
	ASSERT(H5FileIsReadable(h5f) && "ArrayLoad: File is not readable");
	ASSERT(H5DatasetExist(h5f, dataset_name) && "ArrayLoad: Dataset does not exist");

	u32 len;
	H5GetDatasetSize(h5f, dataset_name, &len);
	ASSERT(len == ArrayLen(a) && "ArrayLoad: Array length mismatch");

	H5ReadDatasetf64(h5f, dataset_name, ArrayData(a));
}

void ArraySave(const Array* a, H5FileInfo* h5f, const char* dataset_name) {
	ASSERT(a && h5f && dataset_name && "ArraySave: NULL pointer");
	ASSERT(a->is_host && "ArraySave: Array must be host type");
	ASSERT(H5FileIsWritable(h5f) && "ArraySave: File is not writable");

	H5WriteDatasetf64(h5f, dataset_name, ArrayLen(a), ArrayData(a));
}

__END_DECLS__
