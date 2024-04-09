

#include <metis.h>

#include "common.h"

__BEGIN_DECLS__

static bool u32_2_cmp(const void* a, const void* b) {
	u32* aa = (u32*)a;
	u32* bb = (u32*)b;
	return aa[1] - bb[1];
}

void PartitionMesh3DMETIS(Mesh3DData* mesh, u32 num_part) {
	u32* eind = mesh->ien;
	u32 num_node = mesh->num_node;
	u32 num_tet = mesh->num_tet;
	u32 num_prism = mesh->num_prism;
	u32 num_hex = mesh->num_hex;

	u32* eptr = (u32*)malloc(sizeof(u32) * (num_tet + num_prism + num_hex + 1));

	u32 i;
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

	idx_t* epart = (idx_t*)malloc(sizeof(idx_t) * (num_tet + num_prism + num_hex));
	idx_t* npart = (idx_t*)malloc(sizeof(idx_t) * num_node);

	idx_t objval;
	idx_t ncon = 1;
	idx_t nparts = num_part;

	idx_t options[METIS_NOPTIONS];
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_NUMBERING] = 0;

	METIS_PartMeshNodal(&num_tet, &num_node, eptr, eind, NULL, NULL, &ncon, &nparts, NULL, options, &objval, epart, npart);

	u32 num_elem_ub = num_tet > num_prism ? num_tet : num_prism;
	num_elem_ub = num_elem_ub > num_hex ? num_elem_ub : num_hex;

	u32* buff = (u32*)malloc(sizeof(u32) * num_elem_ub * 2);
	/* Tet */
	/* Indexing and copying */
	for (i = 0; i < num_tet; i++) {
		buff[i * 2] = i;
		buff[i * 2 + 1] = epart[i];
	}
	/* Sort */
	qsort(buff, num_tet, sizeof(u32) * 2, u32_2_cmp);
	/* Reordering */
	for (i = 0; i < num_tet; i++) {

	}

	free(buff);


	free(eptr);
	free(epart);
	free(npart);
}

__END_DECLS__
