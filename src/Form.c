#include "Form.h"

void CDAM_FormCreate(MPI_Comm comm, CDAM_Form **form) {
	*form = (CDAM_Form*)CdamMallocHost(sizeof(CDAM_Form));
	memset(*form, 0, sizeof(CDAM_Form));
	(*form)->comm = comm;
}

void CDAM_FormDestroy(CDAM_Form *form) {
	CdamFreeDevice(form->buffer, form->buffer_size);
	CdamFreeHost(*form);
}

void CDAM_FormSetMesh(CDAM_Form *form, CDAM_Mesh *mesh) {
	form->mesh = mesh;
}

void CDAM_FormGetMesh(CDAM_Form *form, CDAM_Mesh **mesh) {
	if(mesh)
		*mesh = form->mesh;
}

void CDAM_FormConfig(CDAM_Form* form, void* config) {
	form->config = config;
}

void CDAM_FormSetSolution(CDAM_Form *form, value_type *solution) {
	form->solution = solution;
}
void CDAM_FormGetSolution(CDAM_Form *form, value_type **solution) {
	if(solution)
		*solution = form->solution;
}

void CDAM_FormSetContext(CDAM_Form *form, void *context) {
	form->context = context;
}

void CDAM_FormEvalResidual(CDAM_Form *form, CDAM_Vec *b) {
	if(form->residual)
		form->residual(form, b);
}

void CDAM_FormEvalJacobian(CDAM_Form *form, CDAM_Mat *A) {
	if(form->jacobian)
		form->jacobian(form, A);
}

