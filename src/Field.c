

#include <stdlib.h>
#include <string.h>
#include <cublas_v2.h>

#include "alloc.h"
#include "Array.h"
#include "Mesh.h"
#include "Field.h"

__BEGIN_DECLS__

Field* FieldCreate3D(const Mesh3D* mesh, i32 num_nodal_dof) {
	Field* field;

	ASSERT(num_nodal_dof > 0 && "Number of nodal degrees of freedom must be positive.");

	field = (Field*)CdamMallocHost(sizeof(Field));
	memset(field, 0, sizeof(Field));
	u32 num_node = Mesh3DNumNode(mesh);
	field->shape[0] = num_node;
	field->shape[1] = num_nodal_dof;
	
	FieldHost(field) = ArrayCreateHost(num_node * num_nodal_dof);
	FieldDevice(field) = ArrayCreateDevice(num_node * num_nodal_dof);

	return field;
}

void FieldDestroy(Field* f) {
	ArrayDestroy(FieldHost(f));
	FieldHost(f) = NULL;
	ArrayDestroy(FieldDevice(f));
	FieldDevice(f) = NULL;
	free(f);
	f = NULL;
}


void FieldInit(Field* f, void(*func)(f64*, void*), void* ctx) {
	ASSERT(f && func && "FieldInitCond: NULL pointer.");
	Array* host = FieldHost(f);
	(*func)(ArrayData(host), ctx);
	ArrayCopy(FieldDevice(f), FieldHost(f), H2D);
}

void FieldLoad(Field* f, H5FileInfo* h5f, const char* group_name) {
	ASSERT(f && h5f && group_name && "FieldLoad: NULL pointer.");
	ArrayLoad(FieldHost(f), h5f, group_name);
	ArrayCopy(FieldDevice(f), FieldHost(f), H2D);
}

void FieldSave(const Field* f, H5FileInfo* h5f, const char* group_name) {
	/* TODO: Check the consistency of the host and device data. */
	ASSERT(f && h5f && group_name && "FieldSave: NULL pointer.");
	ArraySave(FieldHost(f), h5f, group_name);
}

void FieldCopy(Field* dst, const Field* src) {
	ASSERT(dst && src && "FieldCopy: NULL pointer.");
	/* Further check */
	ArrayCopy(FieldHost(dst), FieldHost(src), H2H);
	ArrayCopy(FieldDevice(dst), FieldDevice(src), D2D);
}

void FieldUpdateHost(Field* f) {
	ASSERT(f && "FieldUpdateHost: NULL pointer.");
	ArrayCopy(FieldHost(f), FieldDevice(f), D2H);
}

void FieldUpdateDevice(Field* f) {
	ASSERT(f && "FieldUpdateDevice: NULL pointer.");
	ArrayCopy(FieldDevice(f), FieldHost(f), H2D);
}

__END_DECLS__
