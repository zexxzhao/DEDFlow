#include <string.h>
#include "alloc.h"
#include "Mesh.h"
#include "commutor.h"

__BEGIN_DECLS__

void AddValuePrivate(void*, void*, index_type, index_type*, index_type);
void CopyValuePrivate(void*, void*, index_type, index_type*, index_type);


void CdamCommutorCreate(MPI_Comm comm, void* mesh, CdamCommutor** commutor) {

	index_type num_node = CdamMeshNumNode((CdamMesh*)mesh);
	index_type* nodal_offset = ((CdamMesh*)mesh)->nodal_offset;
	index_type num_owned_node;
	index_type* l2g = ((CdamMesh*)mesh)->nodal_map_l2g_interior;

	index_type i, j;
	index_type num_fwd_task, num_bwd_task;

	int rank, num_procs;
	int* fwd_count = (*commutor)->fwd_count;
	int* fwd_offset = (*commutor)->fwd_displ;

	int* bwd_count = (*commutor)->bwd_count;
	int* bwd_offset = (*commutor)->bwd_displ;

	int* shared_node = NULL;

	*commutor = CdamTMalloc(CdamCommutor, 1, HOST_MEM);
	memset(*commutor, 0, sizeof(CdamCommutor));
	(*commutor)->comm = comm;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &num_procs);
	num_owned_node = nodal_offset[rank+1] - nodal_offset[rank];

	(*commutor)->rank = rank;
	(*commutor)->num_procs = num_procs;

	/* Count the number of nodes to be sent to each processor */
	j = 0;
	for(i = num_owned_node; i < num_node; ++i) {
		while(j < num_procs && (l2g[i] < nodal_offset[j] || l2g[i] >= nodal_offset[j+1])) {
			++j;
		}
		if(j < num_procs) {
			fwd_count[j]++;
		}
	}

	/* Squeeze out the empty entries and ready the dst, displ */
	fwd_offset[0] = num_owned_node;
	j = 0;
	for(i = 0; i < num_procs; ++i) {
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
	bwd_offset[0] = 0;
	j = 0;
	for(i = 0; i < num_procs; ++i) {
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
	for(i = 0; i < num_procs; ++i) {
		if(fwd_count[i] > 0) {
			fwd_count[j] = fwd_count[i];
			j++;
		}
	}
	memset(fwd_count + j, 0, (num_procs - j) * sizeof(int));
	j = 0;
	for(i = 0; i < num_procs; ++i) {
		if(bwd_count[i] > 0) {
			bwd_count[j] = bwd_count[i];
			j++;
		}
	}
	memset(bwd_count + j, 0, (num_procs - j) * sizeof(int));
	num_bwd_task = (*commutor)->num_bwd_task;
	num_fwd_task = (*commutor)->num_fwd_task;

	/* Allocate the memory for shared_node: the owned nodes that are shared with other processors */
	/* shared_node stores the local indices and is the sendbuff of the bwd task */
	shared_node = CdamTMalloc(int, bwd_offset[num_bwd_task], HOST_MEM);

	/* Fill the shared_node array */
	j = 0;
	for(i = num_owned_node; i < num_node; i++) {
		while(j < num_procs && (l2g[i] < nodal_offset[j] || l2g[i] >= nodal_offset[j+1])) {
			++j;
		}
		shared_node[i] = l2g[i] - nodal_offset[j];
	}

	for(i = 0; i < num_bwd_task; ++i) {
		MPI_Irecv(shared_node + bwd_offset[i], bwd_count[i], MPI_INT,
				      (*commutor)->bwd_srcproc[i], 0, comm, (*commutor)->req + i);
	}

	for(i = 0; i < num_fwd_task; ++i) {
		MPI_Isend(l2g + fwd_offset[i], fwd_count[i], MPI_INT,
				      (*commutor)->fwd_dstproc[i], 0, comm, (*commutor)->req + i + num_bwd_task);
	}

	MPI_Waitall(num_bwd_task + num_fwd_task, (*commutor)->req, MPI_STATUSES_IGNORE);

	(*commutor)->shared_node = shared_node;

}

void CdamCommutorDestroy(CdamCommutor* commutor) {
	int n = commutor->num_bwd_task;
	n = commutor->bwd_displ[n];
	CdamFreeHost(commutor->shared_node, n * sizeof(int));
	CdamFreeHost(commutor, sizeof(CdamCommutor));
}

void CdamCommutorForward(CdamCommutor* commutor, void* sendbuf, index_type blocklen) {
	int i;
	int num_fwd_task = commutor->num_fwd_task;
	int* fwd_count = commutor->fwd_count;
	int* fwd_displ = commutor->fwd_displ;
	int* fwd_dstproc = commutor->fwd_dstproc;
	MPI_Comm comm = commutor->comm;

	int num_bwd_task = commutor->num_bwd_task;
	int* bwd_count = commutor->bwd_count;
	int* bwd_displ = commutor->bwd_displ;
	int* bwd_srcproc = commutor->bwd_srcproc;

	void* ghostbuf = CdamTMalloc(byte, blocklen * fwd_displ[num_fwd_task], HOST_MEM);
	void* ownerbuf = CdamTMalloc(byte, blocklen * bwd_displ[num_bwd_task], HOST_MEM);
	void* ownerbuf_device = CdamTMalloc(byte, blocklen * bwd_displ[num_bwd_task], DEVICE_MEM);

	/* Prefetch the ghost data to the host */
	CdamMemcpy(ghostbuf, (byte*)sendbuf + blocklen * fwd_displ[0],
						 blocklen * (fwd_displ[num_fwd_task] - fwd_displ[0]), HOST_MEM, DEVICE_MEM);

	for(i = 0; i < num_bwd_task; ++i) {
		MPI_Irecv((byte*)ownerbuf + blocklen * bwd_displ[i],
						  blocklen * bwd_count[i], MPI_CHAR,
				      bwd_srcproc[i], 0, comm, commutor->req + i);
	}
	for(i = 0; i < num_fwd_task; ++i) {
		MPI_Isend((byte*)ghostbuf + blocklen * (fwd_displ[i] - fwd_displ[0]),
							blocklen * fwd_count[i], MPI_CHAR,
				      fwd_dstproc[i], 0, comm, commutor->req + i + num_bwd_task);
	}

	MPI_Waitall(num_bwd_task + num_fwd_task, commutor->req, MPI_STATUSES_IGNORE);

	/* Prefetch the owner data to the device */
	CdamMemcpy(ownerbuf_device, ownerbuf, blocklen * bwd_displ[num_bwd_task], DEVICE_MEM, HOST_MEM);

	/* Add to the array */
	/* sendbuf[shared_node[i]] += ownerbuf_device[i] for i in range(bwd_displ[num_bwd_task])*/
	AddValuePrivate(sendbuf, ownerbuf_device, blocklen, commutor->shared_node, bwd_displ[num_bwd_task]);

	CdamFree(ghostbuf, blocklen * fwd_displ[num_fwd_task], HOST_MEM);
	CdamFree(ownerbuf, blocklen * bwd_displ[num_bwd_task], HOST_MEM);
	CdamFree(ownerbuf_device, blocklen * bwd_displ[num_bwd_task], DEVICE_MEM);
}

void CdamCommutorBackword(CdamCommutor* commutor, void* recvbuf, index_type blocklen) {
	int i;
	int num_fwd_task = commutor->num_fwd_task;
	int* fwd_count = commutor->fwd_count;
	int* fwd_displ = commutor->fwd_displ;
	int* fwd_dst = commutor->fwd_dstproc;

	int num_bwd_task = commutor->num_bwd_task;
	int* bwd_count = commutor->bwd_count;
	int* bwd_displ = commutor->bwd_displ;
	int* bwd_src = commutor->bwd_srcproc;
	MPI_Comm comm = commutor->comm;

	void* ghostbuf = CdamTMalloc(byte, blocklen * fwd_displ[num_fwd_task], HOST_MEM);
	void* ownerbuf = CdamTMalloc(byte, blocklen * bwd_displ[num_bwd_task], HOST_MEM);
	void* ownerbuf_device = CdamTMalloc(byte, blocklen * bwd_displ[num_bwd_task], DEVICE_MEM);

	/* Prefetch the data associated with the shared nodes */
	CopyValuePrivate(ownerbuf_device, recvbuf, blocklen, commutor->shared_node, bwd_displ[num_bwd_task]);

	/* Copy the data to the host */
	CdamMemcpy(ownerbuf, ownerbuf_device, blocklen * bwd_displ[num_bwd_task], HOST_MEM, DEVICE_MEM);

	for(i = 0; i < num_fwd_task; ++i) {
		MPI_Irecv(((byte*)ghostbuf) + blocklen * (fwd_displ[i] - fwd_displ[0]),
						  blocklen * fwd_count[i], MPI_CHAR,
				      fwd_dst[i], 0, comm, commutor->req + i);
	}
	for(i = 0; i < num_bwd_task; ++i) {
		MPI_Isend(((byte*)ownerbuf) + blocklen * bwd_displ[i],
							blocklen * bwd_count[i], MPI_CHAR,
				      bwd_src[i], 0, comm, commutor->req + i + num_fwd_task);
	}

	MPI_Waitall(num_bwd_task + num_fwd_task, commutor->req, MPI_STATUSES_IGNORE);

	CdamMemcpy((byte*)recvbuf + blocklen * fwd_displ[0], ghostbuf,
						 blocklen * (fwd_displ[num_fwd_task] - fwd_displ[0]), DEVICE_MEM, HOST_MEM);

	CdamFree(ghostbuf, blocklen * fwd_displ[num_fwd_task], HOST_MEM);
	CdamFree(ownerbuf, blocklen * bwd_displ[num_bwd_task], HOST_MEM);
	CdamFree(ownerbuf_device, blocklen * bwd_displ[num_bwd_task], DEVICE_MEM);
}



__END_DECLS__
