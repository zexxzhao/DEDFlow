#ifndef __FORM_H__
#define __FORM_H__

#include "common.h"

__BEGIN_DECLS__

typedef struct CDAM_Vec CDAM_Vec;
typedef struct CDAM_Mat CDAM_Mat;
typedef struct CDAM_Mesh CDAM_Mesh;
typedef struct CDAM_Form CDAM_Form;

typedef void(*CDAM_FormResidual)(CDAM_Vec* b, void* ctx);
typedef void(*CDAM_FormJacobian)(CDAM_Mat* A, void* ctx);

struct Form {
	MPI_Comm comm;
	CDAM_Mesh *mesh;
	void *config;
	index_type bs;

	u32 buffer_size;
	void* buffer;

	value_type* sol;
	void* ctx;

	CDAM_FormResidual residual;
	CDAM_FormJacobian jacobian;
};

#define CDAM_FormNumComponents(form) ((form)->bs)

void CDAM_FormCreate(MPI_Comm comm, CDAM_Form **form);
void CDAM_FormDestroy(CDAM_Form *form);

void CDAM_FormSetMesh(CDAM_Form *form, CDAM_Mesh *mesh);
void CDAM_FormGetMesh(CDAM_Form *form, CDAM_Mesh **mesh);

void CDAM_FormConfig(CDAM_Form *form, void* config);

void CDAM_FormSetSolution(CDAM_Form *form, value_type *sol);
void CDAM_FormGetSolution(CDAM_Form *form, value_type **sol);

void CDAM_FormSetResidual(CDAM_Form *form, CDAM_FormResidual residual); 
void CDAM_FormSetJacobian(CDAM_Form *form, CDAM_FormJacobian jacobian);

void CDAM_FormEvalResidual(CDAM_Form *form, CDAM_Vec* b);

void CDAM_FormEvalJacobian(CDAM_Form *form, CDAM_Mat* A);


__END_DECLS__

#endif /* __FORM_H__ */
