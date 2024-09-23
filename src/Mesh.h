#ifndef __MESH_H__

#define __MESH_H__


#include <mpi.h>
#include "common.h"
#include "h5util.h"
#include "color.h"

__BEGIN_DECLS__

typedef struct CdamMesh CdamMesh;
struct CdamMesh {
	MPI_Comm comm;

	index_type num[4];
	value_type* coord;
	index_type* ien;

	/* Bound */
	index_type num_bound;
	index_type* bound_id;
	index_type* bound_offset;
	index_type* bound_f2e;
	index_type* bound_forn;
	index_type* bound_node_offset;
	index_type* bound_node;

	/* Partitioned data */
	index_type rank;
	index_type num_procs;

	index_type* nodal_offset;
	index_type num_exclusive_node;

	index_type* nodal_map_l2g_interior; /* Mapping to the global indices;              *
																			 * used for linear systems;                    *
																			 * owned nodes should have consecutive indices */
	index_type* nodal_map_l2g_exterior; /* Mapping to the global indices;              *
																			 * used for output;                            *
																			 *  same as the indices in the mesh file       */

	/* Color */
	index_type num_color;
	index_type* color;
	index_type* color_batch_offset;
	index_type* color_batch_ind;

	/* Batch */
	// index_type num_batch;
	// index_type* batch_offset;
	// index_type* batch_ind;
};

#define CdamMeshNumNode(mesh) (((CdamMesh*)(mesh))->num[0])
#define CdamMeshNumTet(mesh) (((CdamMesh*)(mesh))->num[1])
#define CdamMeshNumPrism(mesh) (((CdamMesh*)(mesh))->num[2])
#define CdamMeshNumHex(mesh) (((CdamMesh*)(mesh))->num[3])
#define CdamMeshNumElem(mesh) (CdamMeshNumTet(mesh) + CdamMeshNumPrism(mesh) + CdamMeshNumHex(mesh))

#define CdamMeshCoord(mesh) (((CdamMesh*)(mesh))->coord)
#define CdamMeshIEN(mesh) (((CdamMesh*)(mesh))->ien)
#define CdamMeshTet(mesh) (CdamMeshIEN(mesh))
#define CdamMeshPrism(mesh) (CdamMeshIEN(mesh) + CdamMeshNumTet(mesh)*4)
#define CdamMeshHex(mesh) (CdamMeshIEN(mesh) + CdamMeshNumTet(mesh)*4 + CdamMeshNumPrism(mesh)*6)

#define CdamMeshNumBound(mesh) (((CdamMesh*)(mesh))->num_bound)
#define CdamMeshBoundID(mesh) (((CdamMesh*)(mesh))->bound_id)
#define CdamMeshBoundOffset(mesh) (((CdamMesh*)(mesh))->bound_offset)
#define CdamMeshBoundF2E(mesh) (((CdamMesh*)(mesh))->bound_f2e)
#define CdamMeshBoundForn(mesh) (((CdamMesh*)(mesh))->bound_forn)

#define CdamMeshBoundNodeOffset(mesh) (((CdamMesh*)(mesh))->bound_node_offset)
#define CdamMeshBoundNumNode(mesh, i) (CdamMeshBoundNodeOffset(mesh)[i+1] - CdamMeshBoundNodeOffset(mesh)[i])
#define CdamMeshBoundNode(mesh, i) (((CdamMesh*)(mesh))->bound_node + CdamMeshBoundNodeOffset(mesh)[i])


#define CdamMeshNumColor(mesh) (((CdamMesh*)(mesh))->num_color)
#define CdamMeshColorBatchOffset(mesh) (((CdamMesh*)(mesh))->color_batch_offset)
#define CdamMeshColorBatchInd(mesh) (((CdamMesh*)(mesh))->color_batch_ind)

#define CdamMeshLocalNodeBegin(mesh) (((CdamMesh*)(mesh))->nodal_offset[mesh->rank])
#define CdamMeshLocalNodeEnd(mesh) (((CdamMesh*)(mesh))->nodal_offset[mesh->rank+1])

/*
 * Expected usage procedure:
 * CdamMesh* mesh;
 * CdamMeshCreate(MPI_COMM_WORLD, &mesh);
 * CdamMeshLoad(mesh, h5handler, "mesh.h5");
 * CdamMeshPrefetch(mesh);
 * CdamMeshColor(mesh);
 * CdamMeshGenerateColorBatch(mesh);
 *
 * ... do something ...
 *
 * CdamMeshDestroy(mesh);
 */
void CdamMeshCreate(MPI_Comm comm, CdamMesh** mesh);
void CdamMeshDestroy(CdamMesh* mesh);
void CdamMeshLoad(CdamMesh* mesh, H5FileInfo* h5f, const char* groupname);
void CdamMeshPrefetch(CdamMesh* mesh);
void CdamMeshColor(CdamMesh* mesh, Arena scratch);
// void CdamMeshGenerateColorBatch(CdamMesh* mesh);


__END_DECLS__

#endif /* __MESH_H__ */


