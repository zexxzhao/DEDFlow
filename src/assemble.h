#ifndef __ASSEMBLE_H__
#define __ASSEMBLE_H__
#include "common.h"



__BEGIN_DECLS__

typedef struct Mesh3D Mesh3D;
typedef struct Field Field;
typedef struct Matrix Matrix;

void AssembleSystemTet(Mesh3D* mesh, f64* wgalpha, f64* dwgalpha, f64* F, Matrix* J);
void AssembleSystemTetFace(Mesh3D* mesh, f64* wgalpha, f64* dwgalpha, f64* F, Matrix* J);


__END_DECLS__

#endif /* __ASSEMBLE_H__ */
