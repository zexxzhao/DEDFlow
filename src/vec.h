#ifndef __VEC_H__
#define __VEC_H__

#include <mpi.h>
#include "common.h"

__BEGIN_DECLS__

enum CDAM_VecType {
	CDAM_VEC_TYPE_NONE = 0x0,
	CDAM_VEC_TYPE_HYPRE = 0x1
};
typedef enum CDAM_VecType CDAM_VecType;

typedef struct CDAM_Vec CDAM_Vec;
struct CDAM_Vec {
	CDAM_VecType type;
	void* vec;
};

CDAM_Vec* VecCreate(MPI_Comm comm, CDAM_VecType type, index_type jlower, index_type jupper);

__END_DECLS__
#endif /* __VEC_H__ */
