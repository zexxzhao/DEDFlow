#include <string.h>
#include "alloc.h"
#include "layout.h"
#include "commu_graph.h"

__BEGIN_DECLS__

#define TAG(d, s) (CDAM_MAX_NUM_TASK * (d) + (s))


void AddValuePrivate(void*, void*, index_type, index_type*, index_type);
void CopyValuePrivate(void*, void*, index_type, index_type*, index_type);


void CommuGraphCreate(MPI_Comm comm, CdamMesh* mesh, CommuGraph** commutor) {

	index_type num_node = CdamMeshNumNode((CdamMesh*)mesh);
	index_type* nodal_offset = ((CdamMesh*)mesh)->nodal_offset;
	index_type num_owned_node = nodal_offset[mesh->rank + 1] - nodal_offset[mesh->rank];
	index_type* l2g = ((CdamMesh*)mesh)->nodal_map_l2g_interior + num_owned_node;

	index_type i, j;
	index_type num_fwd_task, num_bwd_task;
	*commutor = CdamTMalloc(CommuGraph, 1, HOST_MEM);
	CdamMemset(*commutor, 0, sizeof(CommuGraph), HOST_MEM);

	(*commutor)->comm = comm;

	(*commutor)->range_exclusive[0] = 0;
	(*commutor)->range_exclusive[1] = mesh->num_exclusive_node;
	(*commutor)->range_shared[0] = mesh->num_exclusive_node;
	(*commutor)->range_shared[1] = num_owned_node;
	(*commutor)->range_ghosted[0] = num_owned_node;
	(*commutor)->range_ghosted[1] = num_node;


	int* fwd_count = (*commutor)->fwd_count;
	int* fwd_offset = (*commutor)->fwd_displ;

	int* bwd_count = (*commutor)->bwd_count;
	int* bwd_offset = (*commutor)->bwd_displ;


	(*commutor)->rank = mesh->rank;
	(*commutor)->num_procs = mesh->num_procs;

	/* Count the number of nodes to be sent to each processor */
	/* The ghosted nodes are ordered by their owner processors */
	for(i = 0; i < num_node - num_owned_node; ++i) {
		j = 0;
		while(j < mesh->num_procs && (l2g[i] < nodal_offset[j] || l2g[i] >= nodal_offset[j+1])) {
			++j;
		}
		if(j < mesh->num_procs) {
			fwd_count[j]++;
		}
	}

	/* Squeeze out the empty entries and ready the dst, displ */
	// fwd_offset[0] = num_owned_node;
	fwd_offset[0] = 0;
	j = 0;
	for(i = 0; i < mesh->num_procs; ++i) {
		/* If there are nodes to be sent to processor i */
		if(fwd_count[i] > 0) {
			fwd_offset[j + 1] = fwd_offset[j] + fwd_count[i];
			(*commutor)->fwd_dstproc[j] = i;
			j++;
		}
	}

	/* j is the number of processors to which we need to send nodes */
	(*commutor)->num_fwd_task = j;


	/* Count the number of nodes to be received from each processor */
	MPI_Alltoall(fwd_count, 1, MPI_INT, bwd_count, 1, MPI_INT, comm);

	/* Squeeze out the empty entries and ready the src, displ */
	// bwd_offset[0] = mesh->>num_exclusive_node;
	bwd_offset[0] = 0;
	j = 0;
	for(i = 0; i < mesh->num_procs; ++i) {
		/* If there are nodes to be received from processor i */
		if(bwd_count[i] > 0) {
			bwd_offset[j + 1] = bwd_offset[j] + bwd_count[i];
			(*commutor)->bwd_srcproc[j] = i;
			j++;
		}
	}

	/* j is the number of processors from which we need to receive nodes */
	(*commutor)->num_bwd_task = j;


	/* Time to squeeze count */
	j = 0;
	for(i = 0; i < mesh->num_procs; ++i) {
		if(fwd_count[i] > 0) {
			fwd_count[j] = fwd_count[i];
			j++;
		}
	}
	memset(fwd_count + j, 0, (mesh->num_procs - j) * sizeof(int));
	j = 0;
	for(i = 0; i < mesh->num_procs; ++i) {
		if(bwd_count[i] > 0) {
			bwd_count[j] = bwd_count[i];
			j++;
		}
	}
	memset(bwd_count + j, 0, (mesh->num_procs - j) * sizeof(int));
	num_bwd_task = (*commutor)->num_bwd_task;
	num_fwd_task = (*commutor)->num_fwd_task;

	/* Allocate the memory for bwd_task_index: the owned nodes that are shared with other processors */
	/* bwd_task_index stores the local indices */
	index_type* bwd_task_index = CdamTMalloc(index_type, bwd_offset[num_bwd_task] - bwd_offset[0], HOST_MEM);

	/* Fill the bwd_task_index array */
	for(i = 0; i < num_bwd_task; ++i) {
		MPI_Irecv(bwd_task_index + bwd_offset[i] - bwd_offset[0], bwd_count[i], MPI_INDEX_TYPE,
				      (*commutor)->bwd_srcproc[i], TAG(mesh->rank, (*commutor)->bwd_srcproc[i]),
							comm, (*commutor)->req + i);
	}

	for(i = 0; i < num_fwd_task; ++i) {
		MPI_Isend(l2g + fwd_offset[i], fwd_count[i], MPI_INDEX_TYPE,
				      (*commutor)->fwd_dstproc[i], TAG((*commutor)->fwd_dstproc[i], mesh->rank),
							comm, (*commutor)->req + i + num_bwd_task);
	}

	MPI_Waitall(num_bwd_task + num_fwd_task, (*commutor)->req, MPI_STATUSES_IGNORE);
	for(i = 0; i < bwd_offset[num_bwd_task] - bwd_offset[0]; ++i) {
		ASSERT(nodal_offset[mesh->rank] <= bwd_task_index[i] && bwd_task_index[i] < nodal_offset[mesh->rank+1] && "Invalid shared node index");
		bwd_task_index[i] -= nodal_offset[mesh->rank];
	}

	(*commutor)->bwd_task_index = bwd_task_index;

}

void CommuGraphDestroy(CommuGraph* commutor) {
	int n = commutor->num_bwd_task;
	n = commutor->bwd_displ[n] - commutor->bwd_displ[0];
	CdamFree(commutor->bwd_task_index, n * sizeof(index_type), HOST_MEM);
	CdamFree(commutor, sizeof(CommuGraph), HOST_MEM);
}

#ifdef COMMU_GRAPH_USE_ANREA
void CommuGraphSyncForward(CommuGraph* commutor, value_type* sendbuf, size_t blocklen, Arena scratch) {
#else
void CommuGraphSyncForward(CommuGraph* commutor, value_type* sendbuf, size_t blocklen) {
#endif
	int i, j;
	MPI_Comm comm = commutor->comm;
	int num_fwd_task = commutor->num_fwd_task;
	int* fwd_count = commutor->fwd_count;
	int* fwd_displ = commutor->fwd_displ;
	int* fwd_dstproc = commutor->fwd_dstproc;

	int num_bwd_task = commutor->num_bwd_task;
	int* bwd_count = commutor->bwd_count;
	int* bwd_displ = commutor->bwd_displ;
	int* bwd_srcproc = commutor->bwd_srcproc;

	value_type* seg_ghosted = sendbuf + blocklen * commutor->range_ghosted[0];
	index_type len_seg_ghosted = commutor->range_ghosted[1] - commutor->range_ghosted[0];
	value_type* seg_shared = sendbuf + blocklen * commutor->range_shared[0];
	index_type len_seg_shared = commutor->range_shared[1] - commutor->range_shared[0];

#ifdef COMMU_GRAPH_USE_ANREA
	value_type* ghostbuf = (value_type*)AllocInArena(sizeof(value_type) * blocklen,
																									 len_seg_ghosted,
																									 &scratch, ARENA_ON_HOST);
	value_type* ownerbuf = (value_type*)AllocInArena(sizeof(value_type) * blocklen,
																									 len_seg_shared,
																									 &scratch, ARENA_ON_HOST);
	value_type* ownerbuf_recv = (value_type*)AllocInArena(sizeof(value_type) * blocklen,
																												bwd_displ[num_bwd_task] - bwd_displ[0],
																												&scratch, ARENA_ON_HOST);
#else
	value_type* ghostbuf = CdamTMalloc(value_type, blocklen * len_seg_ghosted, HOST_MEM);
	value_type* ownerbuf = CdamTMalloc(value_type, blocklen * len_seg_shared, HOST_MEM);
	value_type* ownerbuf_recv = CdamTMalloc(value_type, blocklen * (bwd_displ[num_bwd_task] - bwd_displ[0]), HOST_MEM);
#endif

	/* Prefetch the ghost data to the host */
	CdamMemcpy(ghostbuf, seg_ghosted, sizeof(value_type) * blocklen * len_seg_ghosted, HOST_MEM, DEVICE_MEM);
	CdamMemcpy(ownerbuf, seg_shared, sizeof(value_type) * blocklen * len_seg_shared, HOST_MEM, DEVICE_MEM);


	for(i = 0; i < num_bwd_task; ++i) {
		MPI_Irecv(ownerbuf_recv + blocklen * (bwd_displ[i] - bwd_displ[0]),
						  (int)(blocklen * bwd_count[i]), MPI_VALUE_TYPE,
				      bwd_srcproc[i], 0, comm, commutor->req + i);
	}
	for(i = 0; i < num_fwd_task; ++i) {
		MPI_Isend(ghostbuf + blocklen * (fwd_displ[i] - fwd_displ[0]),
							(int)(blocklen * fwd_count[i]), MPI_VALUE_TYPE,
				      fwd_dstproc[i], 0, comm, commutor->req + i + num_bwd_task);
	}

	MPI_Waitall(num_bwd_task + num_fwd_task, commutor->req, MPI_STATUSES_IGNORE);


	/* Add to the array */
	/* sendbuf[bwd_task_index[i]] += ownerbuf_device[i] for i in range(bwd_displ[num_bwd_task])*/
	for(i = 0; i < bwd_displ[num_bwd_task] - bwd_displ[0]; ++i) {
		for(j = 0; j < (int)blocklen; ++j) {
			ownerbuf[(commutor->bwd_task_index[i] - commutor->range_shared[0]) * blocklen + j] += ownerbuf_recv[i * blocklen + j];
		}
	}
	
	CdamMemcpy(seg_shared, ownerbuf,
						 blocklen * (commutor->range_shared[1] - commutor->range_shared[0]) * sizeof(value_type),
						 DEVICE_MEM, HOST_MEM);
#ifndef COMMU_GRAPH_USE_ANREA
	CdamFree(ghostbuf, blocklen * len_seg_ghosted * sizeof(value_type), HOST_MEM);
	CdamFree(ownerbuf, blocklen * len_seg_shared * sizeof(value_type), HOST_MEM);
	CdamFree(ownerbuf_recv, blocklen * (bwd_displ[num_bwd_task] - bwd_displ[0]) * sizeof(value_type), HOST_MEM);
#endif
}

#ifdef COMMU_GRAPH_USE_ANREA
void CommuGraphSyncBackward(CommuGraph* commutor, value_type* recvbuf, size_t blocklen, Arena scratch) {
#else
void CommuGraphSyncBackward(CommuGraph* commutor, value_type* recvbuf, size_t blocklen) {
#endif
	int i, j;

	MPI_Comm comm = commutor->comm;
	int num_fwd_task = commutor->num_fwd_task;
	int* fwd_count = commutor->fwd_count;
	int* fwd_displ = commutor->fwd_displ;
	int* fwd_dst = commutor->fwd_dstproc;

	int num_bwd_task = commutor->num_bwd_task;
	int* bwd_count = commutor->bwd_count;
	int* bwd_displ = commutor->bwd_displ;
	int* bwd_src = commutor->bwd_srcproc;

	value_type* seg_ghosted = recvbuf + blocklen * commutor->range_ghosted[0];
	index_type len_seg_ghosted = commutor->range_ghosted[1] - commutor->range_ghosted[0];
	value_type* seg_shared = recvbuf + blocklen * commutor->range_shared[0];
	index_type len_seg_shared = commutor->range_shared[1] - commutor->range_shared[0];

#ifdef COMMU_GRAPH_USE_ANREA
	value_type* ghostbuf = (value_type*)AllocInArena(sizeof(value_type) * blocklen,
																									 len_seg_ghosted, &scratch, ARENA_ON_HOST);
	value_type* ownerbuf = (value_type*)AllocInArena(sizeof(value_type) * blocklen,
																									 len_seg_shared, &scratch, ARENA_ON_HOST);
	value_type* ownerbuf_send = (value_type*)AllocInArena(sizeof(value_type) * blocklen,
																												bwd_displ[num_bwd_task] - bwd_displ[0], &scratch, ARENA_ON_HOST);
#else
	value_type* ghostbuf = CdamTMalloc(value_type, blocklen * len_seg_ghosted, HOST_MEM);
	value_type* ownerbuf = CdamTMalloc(value_type, blocklen * len_seg_shared, HOST_MEM);
	value_type* ownerbuf_send = CdamTMalloc(value_type, blocklen * (bwd_displ[num_bwd_task] - bwd_displ[0]), HOST_MEM);
#endif

	/* Copy the data to the host */
	CdamMemcpy(ownerbuf, seg_shared, sizeof(value_type) * blocklen * len_seg_shared, HOST_MEM, DEVICE_MEM);

	/* Reorder into ownerbuf_send */
	for(i = 0; i < bwd_displ[num_bwd_task] - bwd_displ[0]; ++i) {
		for(j = 0; j < (int)blocklen; ++j) {
			ownerbuf_send[i * blocklen + j] = ownerbuf[(commutor->bwd_task_index[i] - commutor->range_shared[0]) * blocklen + j];
		}
	}

	for(i = 0; i < num_fwd_task; ++i) {
		MPI_Irecv(ghostbuf + blocklen * (fwd_displ[i] - fwd_displ[0]),
						  (int)(blocklen * fwd_count[i]), MPI_VALUE_TYPE,
				      fwd_dst[i], 0, comm, commutor->req + i);
	}
	for(i = 0; i < num_bwd_task; ++i) {
		MPI_Isend(ownerbuf_send + blocklen * (bwd_displ[i] - bwd_displ[0]),
							(int)(blocklen * bwd_count[i]), MPI_VALUE_TYPE,
				      bwd_src[i], 0, comm, commutor->req + i + num_fwd_task);
	}

	MPI_Waitall(num_bwd_task + num_fwd_task, commutor->req, MPI_STATUSES_IGNORE);

	CdamMemcpy(seg_ghosted, ghostbuf,
						 blocklen * len_seg_ghosted * sizeof(value_type),
						 DEVICE_MEM, HOST_MEM);

#ifndef COMMU_GRAPH_USE_ANREA
	CdamFree(ghostbuf, blocklen * len_seg_ghosted * sizeof(value_type), HOST_MEM);
	CdamFree(ownerbuf, blocklen * len_seg_shared * sizeof(value_type), HOST_MEM);
	CdamFree(ownerbuf_send, blocklen * (bwd_displ[num_bwd_task] - bwd_displ[0]) * sizeof(value_type), HOST_MEM);
#endif
}

/*
void CdamCommuForward(* commutor, void* sendbuf, void* layout, size_t blocklen) {
	CdamLayout* map = (CdamLayout*)layout;

	index_type* offset = CdamLayoutComponentOffset(map);
	index_type n = CdamLayoutNumComponent(map);
	index_type nnode = CdamLayoutNumNode(map);

	index_type i, bs;
	byte* buf = (byte*)sendbuf;

	for(i = 0; i < n; ++i) {
		bs = offset[i+1] - offset[i];
		Forward(commutor, buf, blocklen * bs);
		buf += nnode * bs * blocklen;
	}
}
void CdamCommuBackward(* commutor, void* recvbuf, void* layout, size_t blocklen) {
	CdamLayout* map = (CdamLayout*)layout;

	index_type* offset = CdamLayoutComponentOffset(map);
	index_type n = CdamLayoutNumComponent(map);
	index_type nnode = CdamLayoutNumNode(map);

	index_type i, bs;
	byte* buf = (byte*)recvbuf;

	for(i = 0; i < n; ++i) {
		bs = offset[i+1] - offset[i];
		Backword(commutor, buf, blocklen * bs);
		buf += nnode * bs * blocklen;
	}
}
*/




__END_DECLS__
