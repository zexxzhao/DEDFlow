#ifndef __PARTITION_H__
#define __PARTITION_H__


#include "common.h"

#include "Mesh.h"
__BEGIN_DECLS__

void PartitionMesh3DMETIS(Mesh3D* mesh, u32 num_part);

__END_DECLS__

#endif /* __PARTITION_H__ */