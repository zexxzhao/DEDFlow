#ifndef __COMMU_GRAPH_H__
#define __COMMU_GRAPH_H__

#include <mpi.h>
#include "common.h"
#include "Mesh.h"

__BEGIN_DECLS__

#define CDAM_COMMUTOR_MAX_NUM_TASK (1024)

/* The interprocessor mapping is defined by the user */
/* The vec consists of three parts: exclusive, shared, ghosted */
/* Forward communication: add ghosted to shared */
/* Backward communication: copy shared to ghosted */
/* Consider the following inter processor communication pattern
 * Node:  Exclusive | Shared | Ghosted
 * P0:    0 1       | 2 3 4  |
 * P1:    0 1 2 3   |        | 4 5
 * P2:    0 1 2 3   |        | 4 5 6
 * And the ghost-to-shared communication pattern is
 * P1: 4 5 -> P0: 2 4
 * P2: 4 5 6 -> P0: 2 3 4
 *
 * The forward task indices are alway contiguous, while the backward task indices are not.
 * For P0:
 * num_fwd_task = 0
 * num_bwd_task = 2
 * bwd_count = [2, 3]
 * bwd_displ = [0, 2]
 * bwd_task_index = [2 4 2 3 4]
 *
 * For P1:
 * num_fwd_task = 1
 * fwd_count = [2]
 * fwd_displ = [0]
 * num_bwd_task = 0
 * 
 * For P2:
 * num_fwd_task = 1
 * fwd_count = [3]
 * fwd_displ = [0]
 * num_bwd_task = 0
 *
 */

struct CommuGraph {
	MPI_Comm comm;
	int rank, num_procs;

	int num_fwd_task;
	int fwd_dstproc[CDAM_COMMUTOR_MAX_NUM_TASK]; /* connected part */
	int fwd_count[CDAM_COMMUTOR_MAX_NUM_TASK]; /* forward size */
	int fwd_displ[CDAM_COMMUTOR_MAX_NUM_TASK]; /* forward displacement */

	int num_bwd_task;
	int bwd_srcproc[CDAM_COMMUTOR_MAX_NUM_TASK]; /* connected part */
	int bwd_count[CDAM_COMMUTOR_MAX_NUM_TASK]; /* backward size */
	int bwd_displ[CDAM_COMMUTOR_MAX_NUM_TASK]; /* backward displacement */

	MPI_Request req[CDAM_COMMUTOR_MAX_NUM_TASK * 2]; /* request for communication */
	MPI_Status stat[CDAM_COMMUTOR_MAX_NUM_TASK * 2]; /* status for communication */

	index_type* bwd_task_index;
	int range_exclusive[2];
	int range_shared[2];
	int range_ghosted[2];
};
typedef struct CommuGraph CommuGraph;


void CommuGraphCreate(MPI_Comm comm, CdamMesh* mesh, CommuGraph** commutor);
void CommuGraphDestroy(CommuGraph* commutor);

#ifdef COMMU_GRAPH_USE_ARENA
void CommuGraphSyncForward(CommuGraph* commu, value_type* vec, size_t blocklen, Arena scratch);
void CommuGraphSyncBackward(CommuGraph* commu, value_type* vec, size_t blocklen, Arena scratch);
#else
void CommuGraphSyncForward(CommuGraph* commu, value_type* vec, size_t blocklen);
void CommuGraphSyncBackward(CommuGraph* commu, value_type* vec, size_t blocklen);
#endif



__END_DECLS__


#endif /* __COMMU_GRAPH_H__ */
