#ifndef __COLOR_H__
#define __COLOR_H__

#include "common.h"
#define UNCOLORED (0x0)
#define MAX_COLOR (1 << 8)


__BEGIN_DECLS__

typedef struct Mesh3D Mesh3D;
typedef i32 color_t;
void ColorMeshTet(const Mesh3D* mesh, index_type max_color_len, color_t* color);
void ColorMeshPrism(const Mesh3D* mesh, index_type max_color_len, color_t* color);
void ColorMeshHex(const Mesh3D* mesh, index_type max_color_len, color_t* color);

color_t GetMaxColor(const color_t* color, index_type num_elem);
__END_DECLS__

#endif /* __COLOR_H__ */
