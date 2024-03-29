#ifndef __FIELD_H__
#define __FIELD_H__

#include "common.h"
// #include "Array.h"
#include "Mesh.h"

__BEGIN_DECLS__

typedef struct Array Array;

typedef struct Field Field;
struct Field {
	u32 shape[2];
	Array* host;
	Array* device;
};

#define FieldHost(f) ((f)->host)
#define FieldDevice(f) ((f)->device)

Field* FieldCreate3D(const Mesh3D* mesh, i32 num_nodal_dof);
void FieldDestroy(Field* f);

// void FieldInitCond(Field* f, void(*func)(f64*, const f64*, const void* ctx), const void* ctx);

void FieldLoad(Field* f, H5FileInfo* h5f, const char* group_name);
void FieldSave(const Field* f, H5FileInfo* h5f, const char* group_name);

void FieldCopy(Field* dst, const Field* src);

void FieldUpdateHost(Field* f);
void FieldUpdateDevice(Field* f);

__END_DECLS__

#endif /* __FIELD_H__ */
