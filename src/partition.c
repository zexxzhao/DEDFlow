

#include <metis.h>

#include "common.h"

__BEGIN_DECLS__

static bool u32_2_cmp(const void* a, const void* b) {
	u32* aa = (u32*)a;
	u32* bb = (u32*)b;
	return aa[1] - bb[1];
}

void PartitionMesh3DMETIS(Mesh3D* mesh, u32 num_part) {
	Mesh3DData* host = Mesh3DHost(mesh);
	u32* eind = Mesh3DDataIEN(host);
	u32 num_node = Mesh3DNumNode(mesh);
	u32 num_tet = Mesh3DNumTet(mesh);
	u32 num_prism = Mesh3DNumPrism(mesh);
	u32 num_hex = Mesh3DNumHex(mesh);

	u32* eptr;
	idx_t* epart, npart;

	u32 i;

	/* Generate the offset array */
	eptr = (u32*)CdamMallocHost(sizeof(u32) * (num_tet + num_prism + num_hex + 1));
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
	METIS_PartMeshNodal(&num_tet, &num_node, eptr, eind, NULL, NULL, &ncon, &nparts, NULL, options, &objval, epart, npart);

	/* Copy the partitioning result */
	mesh->num_part = num_part;
	for (i = 0; i < num_tet + num_prism + num_hex; i++) {
		mesh->epart[i] = epart[i];
	}

	CdamFreeHost(eptr, sizeof(u32) * (num_tet + num_prism + num_hex + 1));
	CdamFreeHost(epart, sizeof(idx_t) * (num_tet + num_prism + num_hex));
	CdamFreeHost(npart, sizeof(idx_t) * num_node);
}

__END_DECLS__
