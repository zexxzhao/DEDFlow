

#include <string.h>

#ifdef USE_METIS
#include <metis.h>
#else
typedef int idx_t;
#endif

#include "alloc.h"
#include "Mesh.h"

__BEGIN_DECLS__

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
	idx_t* eind = (idx_t*)CdamMallocHost(SIZE_OF(idx_t) * (num_tet * 4 + num_prism * 6 + num_hex * 8));
	memcpy(eind, ien, SIZE_OF(idx_t) * (num_tet * 4 + num_prism * 6 + num_hex * 8));
	/* Generate the offset array */
	eptr = (idx_t*)CdamMallocHost(SIZE_OF(idx_t) * (num_tet + num_prism + num_hex + 1));
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
	epart = (idx_t*)CdamMallocHost(SIZE_OF(idx_t) * (num_tet + num_prism + num_hex));
	npart = (idx_t*)CdamMallocHost(SIZE_OF(idx_t) * num_node);

	/* Prepare the METIS input */
	idx_t objval;
	idx_t ncon = 1;
	idx_t nparts = num_part;

	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_NUMBERING] = 0;

	/* Call METIS */
	METIS_PartMeshNodal(&num_tet, &num_node, eptr, eind, NULL, NULL, &nparts, NULL, options, &objval, epart, npart);
	epart_index_type = (index_type*)CdamMallocHost(SIZE_OF(index_type) * (num_tet + num_prism + num_hex));
	for (i = 0; i < num_tet + num_prism + num_hex; i++) {
		epart_index_type[i] = (index_type)epart[i];
	}

	/* Copy the partitioning result */
	cudaMemcpy(mesh->epart, epart_index_type, SIZE_OF(index_type) * (num_tet + num_prism + num_hex), cudaMemcpyHostToDevice);

	CdamFreeHost(epart_index_type, SIZE_OF(index_type) * (num_tet + num_prism + num_hex));
	CdamFreeHost(eind, SIZE_OF(idx_t) * (num_tet * 4 + num_prism * 6 + num_hex * 8));
	CdamFreeHost(eptr, SIZE_OF(index_type) * (num_tet + num_prism + num_hex + 1));
	CdamFreeHost(epart, SIZE_OF(idx_t) * (num_tet + num_prism + num_hex));
	CdamFreeHost(npart, SIZE_OF(idx_t) * num_node);
#endif
}

__END_DECLS__
