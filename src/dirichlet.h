#ifndef __DIRICHLET_H__
#define __DIRICHLET_H__

#include "common.h"
#include "Mesh.h"
#include "matrix.h"

__BEGIN_DECLS__

enum BCType {
	BC_NONE = 0,
	BC_STRONG = 1,
	BC_WEAK = 2,
	BC_OUTFLOW = 4
};
typedef enum BCType BCType;

typedef struct Dirichlet Dirichlet;
struct Dirichlet {
	const CdamMesh* mesh;
	index_type face_ind;
	index_type shape;
	size_t buffer_size;
	void* buffer;
	BCType bctype[0];
};

Dirichlet* DirichletCreate(const CdamMesh* mesh, index_type face_ind, index_type shape);
void DirichletDestroy(Dirichlet* dirichlet);

void DirichletApplyVec(Dirichlet* dirichlet, value_type* b);
void DirichletApplyMat(Dirichlet* dirichlet, Matrix* A);


__END_DECLS__

#endif /* __DIRICHLET_H__ */	
