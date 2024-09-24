#ifndef __COMMUTOR_H__
#define __COMMUTOR_H__

#include <mpi.h>
#include "common.h"

__BEGIN_DECLS__

#define CDAM_COMMUTOR_MAX_NUM_TASK (1024)

struct CdamCommutor {
	MPI_Comm comm;
	int rank, num_procs;

	int num_fwd_task;
	int fwd_dstproc[CDAM_COMMUTOR_MAX_NUM_TASK]; /* connected part */
	int fwd_count[CDAM_COMMUTOR_MAX_NUM_TASK]; /* forward size */
	int fwd_displ[CDAM_COMMUTOR_MAX_NUM_TASK + 1]; /* forward displacement */

	int num_bwd_task;
	int bwd_srcproc[CDAM_COMMUTOR_MAX_NUM_TASK]; /* connected part */
	int bwd_count[CDAM_COMMUTOR_MAX_NUM_TASK]; /* backward size */
	int bwd_displ[CDAM_COMMUTOR_MAX_NUM_TASK + 1]; /* backward displacement */

	MPI_Request req[CDAM_COMMUTOR_MAX_NUM_TASK]; /* request for communication */
	MPI_Status stat[CDAM_COMMUTOR_MAX_NUM_TASK]; /* status for communication */
	
	int num_node_local;
	int num_node_global;
	int* shared_node;
};
typedef struct CdamCommutor CdamCommutor;


void CdamCommutorCreate(MPI_Comm comm, void* mesh, CdamCommutor** commutor);
void CdamCommutorDestroy(CdamCommutor* commutor);
void CdamCommutorForward(CdamCommutor* commutor, void* sendbuf, index_type blocklen);
void CdamCommutorBackward(CdamCommutor* commutor, void* recvbuf, index_type blocklen);

__END_DECLS__


#endif /* __COMMUTOR_H__ */
