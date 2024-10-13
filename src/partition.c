

#include <string.h>

#ifdef USE_METIS
#include <metis.h>
#else
typedef int idx_t;
#endif

#include "alloc.h"
#include "Mesh.h"

__BEGIN_DECLS__

#ifdef DEBUG
void PartitionMesh3DMETIS(Mesh3D* mesh, index_type num_part) {
#ifdef USE_METIS
	Mesh3DData* host = Mesh3DHost(mesh);
	index_type* ien = Mesh3DDataIEN(host);
	idx_t num_node = Mesh3DNumNode(mesh);
	idx_t num_tet = Mesh3DNumTet(mesh);
	idx_t num_prism = Mesh3DNumPrism(mesh);
	idx_t num_hex = Mesh3DNumHex(mesh);

	idx_t* eptr;
	idx_t* epart;
	idx_t* npart;
	index_type* epart_index_type;

	i32 i;

	/* Generate the element array */
	idx_t* eind = (idx_t*)CdamMallocHost(sizeof(idx_t) * (num_tet * 4 + num_prism * 6 + num_hex * 8));
	memcpy(eind, ien, sizeof(idx_t) * (num_tet * 4 + num_prism * 6 + num_hex * 8));
	/* Generate the offset array */
	eptr = (idx_t*)CdamMallocHost(sizeof(idx_t) * (num_tet + num_prism + num_hex + 1));
	eptr[0] = 0;
	for (i = 0; i < num_tet; i++) {
		eptr[i + 1] = eptr[i] + 4;
	}
	for (i = 0; i < num_prism; i++) {
		eptr[num_tet + i + 1] = eptr[num_tet + i] + 6;
	}
	for (i = 0; i < num_hex; i++) {
		eptr[num_tet + num_prism + i + 1] = eptr[num_tet + num_prism + i] + 8;
	}

	/* Allocate the element and node partition arrays */
	epart = (idx_t*)CdamMallocHost(sizeof(idx_t) * (num_tet + num_prism + num_hex));
	npart = (idx_t*)CdamMallocHost(sizeof(idx_t) * num_node);

	/* Prepare the METIS input */
	idx_t objval;
	idx_t ncon = 1;
	idx_t nparts = num_part;

	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_NUMBERING] = 0;

	/* Call METIS */
	METIS_PartMeshNodal(&num_tet, &num_node, eptr, eind, NULL, NULL, &nparts, NULL, options, &objval, epart, npart);
	epart_index_type = (index_type*)CdamMallocHost(sizeof(index_type) * (num_tet + num_prism + num_hex));
	for (i = 0; i < num_tet + num_prism + num_hex; i++) {
		epart_index_type[i] = (index_type)epart[i];
	}

	/* Copy the partitioning result */
	cudaMemcpy(mesh->epart, epart_index_type, sizeof(index_type) * (num_tet + num_prism + num_hex), cudaMemcpyHostToDevice);

	CdamFreeHost(epart_index_type, sizeof(index_type) * (num_tet + num_prism + num_hex));
	CdamFreeHost(eind, sizeof(idx_t) * (num_tet * 4 + num_prism * 6 + num_hex * 8));
	CdamFreeHost(eptr, sizeof(index_type) * (num_tet + num_prism + num_hex + 1));
	CdamFreeHost(epart, sizeof(idx_t) * (num_tet + num_prism + num_hex));
	CdamFreeHost(npart, sizeof(idx_t) * num_node);
#endif
}
#endif

void PartitionMeshMetis(index_type num[], index_type* ien, index_type num_part,
											  index_type* epart, index_type* npart) {
	ASSERT(num_part > 0);
	if(num_part == 1) {
		CdamMemset(epart, 0, sizeof(index_type) * (num[1] + num[2] + num[3]), HOST_MEM);
		CdamMemset(npart, 0, sizeof(index_type) * num[0], HOST_MEM);
		return;
	}
#ifdef USE_METIS
	index_type i = 0, num_node = num[0];
	idx_t num_tet = num[1];
	idx_t num_prism = num[2];
	idx_t num_hex = num[3];
	idx_t num_elem = num_tet + num_prism + num_hex;

	idx_t* eptr = CdamTMalloc(idx_t, num_elem + 1, HOST_MEM);
	eptr[0] = 0;
	for (i32 i = 0; i < num_tet; i++) {
		eptr[i + 1] = eptr[i] + 4;;
	}
	for (i32 i = 0; i < num_prism; i++) {
		eptr[num_tet + i + 1] = eptr[num_tet + i] + 6;
	}
	for (i32 i = 0; i < num_hex; i++) {
		eptr[num_tet + num_prism + i + 1] = eptr[num_tet + num_prism + i] + 8;
	}

	idx_t* eind = CdamTMalloc(idx_t, num_tet * 4 + num_prism * 6 + num_hex * 8, HOST_MEM);
	for(i = 0; i < num_tet * 4 + num_prism * 6 + num_hex * 8; i++) {
		eind[i] = (idx_t)ien[i];
	}
	idx_t *epart_metis = CdamTMalloc(idx_t, num_elem, HOST_MEM);
	idx_t *npart_metis = CdamTMalloc(idx_t, num_node, HOST_MEM);

	idx_t objval = 1;
	idx_t ncon = 1;
	idx_t nparts = num_part;

	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_NUMBERING] = 0;
	options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
	options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
	options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_METISRB;
	options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
	options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
	options[METIS_OPTION_RTYPE] = METIS_RTYPE_GREEDY;
	options[METIS_OPTION_DBGLVL] = 0;
	options[METIS_OPTION_UFACTOR] = -1;
	options[METIS_OPTION_MINCONN] = 0;
	options[METIS_OPTION_CONTIG] = 0;
	options[METIS_OPTION_SEED] = -1;
	options[METIS_OPTION_NITER] = 10;
	int ret =
	METIS_PartMeshDual(&num_elem, &num_node, eptr, eind, NULL, NULL,
										 &ncon, &nparts, NULL, NULL, &objval,
										 epart_metis, npart_metis);
	ASSERT(ret == METIS_OK && "METIS partitioning failed");

	for(i = 0; i < num_elem; i++) {
		epart[i] = (index_type)epart_metis[i];
	}
	for(i = 0; i < num_node; i++) {
		npart[i] = (index_type)npart_metis[i];
	}

	CdamFree(eptr, sizeof(idx_t) * (num_elem + 1), HOST_MEM);
	CdamFree(eind, sizeof(idx_t) * (num_tet * 4 + num_prism * 6 + num_hex * 8), HOST_MEM);
	CdamFree(epart_metis, sizeof(idx_t) * num_elem, HOST_MEM);
	CdamFree(npart_metis, sizeof(idx_t) * num_node, HOST_MEM);
#endif
}

__END_DECLS__
