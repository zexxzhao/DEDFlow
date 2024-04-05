#include "common.h"



__BEGIN_DECLS__

typedef struct Mesh3D Mesh3D;
typedef struct Field Field;
typedef struct CSRMatrix CSRMatrix;

void assemble_system_tet(Mesh3D* mesh, Field* wgold, Field* dwgold, Field* dwg, f64* F, CSRMatrix* J);


__END_DECLS__
