#ifndef __DIRICHLET_H__
#define __DIRICHLET_H__

#include "common.h"

__BEGIN_DECLS__

enum BCType {
	BC_NONE = 0,
	BC_STRONG = 1,
	BC_WEAK = 2,
	BC_OUTFLOW = 4
};
typedef enum BCType BCType;

typedef struct Mesh3D Mesh3D;
typedef struct Matrix Matrix;

typedef struct Dirichlet Dirichlet;
struct Dirichlet {
	const Mesh3D* mesh;
	index_type face_ind;
	index_type shape;
	size_t buffer_size;
	void* buffer;
	BCType bctype[0];
};

Dirichlet* DirichletCreate(const Mesh3D* mesh, index_type face_ind, index_type shape);
void DirichletDestroy(Dirichlet* dirichlet);

void DirichletApplyVec(Dirichlet* dirichlet, value_type* b);
void DirichletApplyMat(Dirichlet* dirichlet, Matrix* A);


__END_DECLS__

#endif /* __DIRICHLET_H__ */	
