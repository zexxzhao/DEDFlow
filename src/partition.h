#ifndef __PARTITION_H__
#define __PARTITION_H__


#include "common.h"

#include "Mesh.h"
__BEGIN_DECLS__

void PartitionMeshMetis(index_type num[], index_type ien[], index_type num_part,
												index_type* epart, index_type* npart);

__END_DECLS__

#endif /* __PARTITION_H__ */
