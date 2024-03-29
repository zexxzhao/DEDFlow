#ifndef __ARRAY_H__
#define __ARRAY_H__

#include "common.h"

__BEGIN_DECLS__

typedef struct H5FileInfo H5FileInfo;

typedef struct Array Array;
struct Array {
	b32 is_host;
	u32 len;
	f64 *data;
};


Array* ArrayCreateHost(u32 len);
Array* ArrayCreateDevice(u32 len);
void ArrayDestroy(Array* a);

void ArrayCopy(Array* dst, const Array* src, MemCopyKind kind);

void ArraySet(Array* a, f64 val);
void ArrayZero(Array* a);

void ArraySetAt(Array* a, u32 n, const u32* idx, const f64* val);
void ArrayGetAt(const Array* a, u32 n, const u32* idx, f64* val);

void ArrayScale(Array* a, f64 val);
void ArrayDot(f64* result, const Array* a, const Array* b);
void ArrayNorm2(f64* result, const Array* a);

void ArrayAXPY(Array* y, f64 a, const Array* x);
void ArrayAXPBY(Array* y, f64 a, const Array* x, f64 b);

void ArrayLoad(Array* a, H5FileInfo* h5f, const char* dataset_name);
void ArraySave(const Array* a, H5FileInfo* h5f, const char* dataset_name);


#define ArrayLen(a) ((a)->len)
#define ArrayData(a) ((a)->data)

__END_DECLS__

#endif /* __ARRAY_H__ */
