#ifndef __COLOR_H__
#define __COLOR_H__

#include "common.h"
#include "alloc.h"

#define UNCOLORED (0x0)
#define MAX_COLOR (1 << 8)


__BEGIN_DECLS__

struct CdamMesh;
void ColorMeshTet(struct CdamMesh* mesh, index_type max_color_len, index_type* color, Arena scratch);
void ColorMeshPrism(struct CdamMesh* mesh, index_type max_color_len, index_type* color, Arena scratch);
void ColorMeshHex(struct CdamMesh* mesh, index_type max_color_len, index_type* color, Arena scratch);

index_type GetMaxColor(const index_type* color, index_type num_elem);
__END_DECLS__

#endif /* __COLOR_H__ */
