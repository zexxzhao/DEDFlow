#include <string.h>
#include <HYPRE.h>

int ffs(int);
#include <_hypre_utilities.h>
#include <_hypre_IJ_mv.h>

#include "alloc.h"
#include "Mesh.h"
#include "json.h"
#include "vec.h"

__BEGIN_DECLS__


void CdamVecCreate(MPI_Comm comm, void* mesh, CdamVec** vec) {
	CdamMesh* mesh3d = (CdamMesh*)mesh;
	*vec = (CdamVec*)CdamMallocHost(sizeof(CdamVec));
	memset(*vec, 0, sizeof(CdamVec));
	CdamVecComm(*vec) = comm;

	CdamVecPartitionLower(*vec) = CdamMeshLocalNodeBegin(mesh3d);
	CdamVecPartitionUpper(*vec) = CdamMeshLocalNodeEnd(mesh3d);
}

void CdamVecSetNumComponents(CdamVec* vec, index_type num_components) {
	CdamVecNumComponents(vec) = num_components;
}

void CdamVecSetComponentSize(CdamVec* vec, index_type component, index_type size) {
	CdamVecComponentSize(vec)[component] = size;
}

void CdamVecInitialize(CdamVec* vec) {
	index_type i, bs;
	index_type num_components = CdamVecNumComponents(vec);
	index_type jlower = CdamVecPartitionLower(vec);
	index_type jupper = CdamVecPartitionUpper(vec);
	MPI_Comm comm = CdamVecComm(vec);
	HYPRE_IJVector subvec;
	HYPRE_MemoryLocation mem_location;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
	mem_location = HYPRE_MEMORY_DEVICE;
#else
	mem_location = HYPRE_MEMORY_HOST;
#endif

	index_type* offset = (index_type*)CdamMallocHost((num_components + 1) * sizeof(index_type));
	offset[0] = 0;
	for(i = 0; i < num_components; i++) {
		offset[i + 1] = offset[i] + CdamVecComponentSize(vec)[i];
	}
	value_type* buffer = hypre_CTAlloc(value_type, offset[num_components], mem_location);

	for(i = 0; i < CdamVecNumComponents(vec); i++) {
		bs = CdamVecComponentSize(vec)[i];
		subvec = CdamVecSub(vec)[i];
		HYPRE_IJVectorCreate(comm, jlower * bs, jupper * bs, &subvec);
		HYPRE_IJVectorSetObjectType(subvec, HYPRE_PARCSR);
		HYPRE_IJVectorInitialize(subvec);
		ASSERT(hypre_IJVectorMemoryLocation(CdamVecSub(vec)[i]) == mem_location && "Memory location mismatch.\n");

		/* Free the old buffer and allocate a new one to ensure contiguous memory */
		hypre_TFree(CdamVecRawPtr(vec, i), mem_location);
		CdamVecRawPtr(vec, i) =  buffer + offset[i] * (jupper - jlower);
	}

	CdamFreeHost(offset, (num_components + 1) * sizeof(index_type));
}

void CdamVecDestroy(CdamVec* vec) {
	index_type i;
	/* Retrieve the buffer pointer to avoid mem leak */
	value_type* buffer = CdamVecRawPtr(vec, 0);
	HYPRE_MemoryLocation mem_location;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
	mem_location = HYPRE_MEMORY_DEVICE;
#else
	mem_location = HYPRE_MEMORY_HOST;
#endif
	for(i = 0; i < CdamVecNumComponents(vec); i++) {
		/* Set the buffer pointer to NULL to avoid double-free */
		CdamVecRawPtr(vec, i) = NULL;
		HYPRE_IJVectorDestroy(CdamVecSub(vec)[i]);
	}
	CdamFreeHost(vec, sizeof(CdamVec));
	hypre_TFree(buffer, mem_location); 
}

void CdamVecGetArray(CdamVec* vec, value_type** array) {
	index_type i;
	*array = CdamVecRawPtr(vec, 0);
	for(i = 0; i < CdamVecNumComponents(vec); i++) {
		CdamVecRawPtr(vec, i) = NULL;
	}
}
void CdamVecRestoreArray(CdamVec* vec, value_type** array) {
	index_type i, bs;
	index_type jlower = CdamVecPartitionLower(vec);
	index_type jupper = CdamVecPartitionUpper(vec);
	for(i = 0; i < CdamVecNumComponents(vec); i++) {
		bs = CdamVecComponentSize(vec)[i];
		CdamVecRawPtr(vec, i) = *array + bs * (jupper - jlower);
	}
	*array = NULL;
}

void CdamVecCopy(CdamVec* src, CdamVec* dest) {
	value_type* src_array, *dest_array;
	index_type len;
	index_type i;
	HYPRE_MemoryLocation mem_location;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
	mem_location = HYPRE_MEMORY_DEVICE;
#else
	mem_location = HYPRE_MEMORY_HOST;
#endif

	len = 0;
	for(i = 0; i < CdamVecNumComponents(src); i++) {
		len += CdamVecComponentSize(src)[i];
	}
	len *= (CdamVecPartitionUpper(src) - CdamVecPartitionLower(src));

	CdamVecGetArray(src, &src_array);
	CdamVecGetArray(dest, &dest_array);

	hypre_TMemcpy(dest_array, src_array, value_type, len, mem_location, mem_location);

	CdamVecRestoreArray(src, &src_array);
	CdamVecRestoreArray(dest, &dest_array);
}

void CdamVecAssemble(CdamVec* vec) {
	index_type i;
	for(i = 0; i < CdamVecNumComponents(vec); i++) {
		HYPRE_IJVectorAssemble(CdamVecSub(vec)[i]);
	}
}

void CdamVecAXPY(value_type alpha, CdamVec* x, CdamVec* y) {
	index_type i;
	for(i = 0; i < CdamVecNumComponents(x); i++) {
		hypre_SeqVectorAxpy(alpha, CdamVecSubSeqVec(x, i), CdamVecSubSeqVec(y, i));
	}
}

void CdamVecScale(CdamVec* x, value_type alpha) {
	index_type i;
	for(i = 0; i < CdamVecNumComponents(x); i++) {
		hypre_SeqVectorScale(alpha, CdamVecSubSeqVec(x, i));
	}
}

void CdamVecZero(CdamVec* x) {
	index_type i;
	for(i = 0; i < CdamVecNumComponents(x); i++) {
		hypre_SeqVectorSetConstantValues(CdamVecSubSeqVec(x, i), (value_type)0.0);
	}
}

void CdamVecInnerProd(CdamVec* x, CdamVec* y, value_type* result, value_type* separated_results) {
	index_type i;
	value_type local_result = 0.0;
	value_type local_separated_results[CDAM_VEC_MAX_NUM_COMPONENTS];
	for(i = 0; i < CdamVecNumComponents(x); i++) {
		local_separated_results[i] = hypre_SeqVectorInnerProd(CdamVecSubSeqVec(x, i), CdamVecSubSeqVec(y, i));
		local_result += local_separated_results[i];
	}
	if(result) {
		hypre_MPI_Allreduce(&local_result, result, 1, HYPRE_MPI_REAL,
												 hypre_MPI_SUM, CdamVecComm(x));
	}
	if(separated_results) {
		hypre_MPI_Allreduce(local_separated_results, separated_results, 
												CdamVecNumComponents(x), HYPRE_MPI_REAL,
												hypre_MPI_SUM, CdamVecComm(x));
	}
}

void CdamVecNorm(CdamVec* x, value_type* result, value_type* separated_results) {
	index_type i;
	CdamVecInnerProd(x, x, result, separated_results);
	if(result) {
		*result = sqrt(*result);
	}
	if(separated_results) {
		for(i = 0; i < CdamVecNumComponents(x); i++) {
			separated_results[i] = sqrt(separated_results[i]);
		}
	}
}

void CdamVecView(CdamVec* x, FILE* ostream) {}

void CdamVecLayoutCreate(CdamVecLayout** layout, void* config) {
	*layout = CdamTMalloc(CdamVecLayout, 1, HOST_MEM);
	CdamMemset(*layout, 0, sizeof(CdamVecLayout), HOST_MEM);
	cJSON* json = (cJSON*)config;
	b32 has_ns = JSONGetItem(json, "VMS.IncompressibleNS.Included")->valuedouble > 0.0;
	b32 has_t = JSONGetItem(json, "VMS.Temperature.Included")->valuedouble > 0.0;
	b32 has_phi = JSONGetItem(json, "VMS.Levelset.Included")->valuedouble > 0.0;

	index_type num_components = 0;

	if(has_ns) {
		num_components++;
		CdamVecLayoutComponentOffset(*layout)[num_components] = CdamVecLayoutComponentOffset(*layout)[num_components - 1] + 4;
	}

	if(has_t) {
		num_components++;
		CdamVecLayoutComponentOffset(*layout)[num_components] = CdamVecLayoutComponentOffset(*layout)[num_components - 1] + 1;
	}

	if(has_phi) {
		num_components++;
		CdamVecLayoutComponentOffset(*layout)[num_components] = CdamVecLayoutComponentOffset(*layout)[num_components - 1] + 1;
	}

	CdamVecLayoutNumComponents(*layout) = num_components;

}

void CdamVecLayoutDestroy(CdamVecLayout* layout) {
	CdamFree(layout, sizeof(CdamVecLayout), HOST_MEM);
}

void CdamVecLayoutSetup(CdamVecLayout* layout, void* mesh) {
	CdamMesh* mesh3d = (CdamMesh*)mesh;
	CdamVecLayoutNumOwned(layout) = CdamMeshLocalNodeEnd(mesh3d) - CdamMeshLocalNodeBegin(mesh3d);
	CdamVecLayoutNumGhost(layout) = CdamMeshNumNode(mesh3d) - CdamVecLayoutNumOwned(layout);
}

void VecDot(void* x, void* y, void* layout, void* result, void* results) {
	value_type* vx = (value_type*)x;
	value_type* vy = (value_type*)y;
	CdamVecLayout* lo = (CdamVecLayout*)layout;
	value_type* r = (value_type*)result;
	value_type* rs = (value_type*)results;

	value_type local_result[CDAM_VEC_MAX_NUM_COMPONENTS];

	index_type i, n = CdamVecLayoutNumComponents(lo);
	index_type num_owned = CdamVecLayoutNumOwned(lo);
	index_type num_ghost = CdamVecLayoutNumGhost(lo);
	index_type num = num_owned + num_ghost;
	index_type begin, end;
	for(i = 0; i < n; i++) {
		local_result[i] = 0.0;
		begin = CdamVecLayoutComponentOffset(lo)[i];
		end = CdamVecLayoutComponentOffset(lo)[i + 1];
		BLAS_CALL(dot, CdamLayoutNumOwned(lo) * (end - begin),
							vx + begin * num, 1,
							vy + begin * num, 1, local_result);																
	}
	MPI_Allreduce(MPI_IN_PLACE, local_result, n, MPI_VALUE_TYPE, MPI_SUM, MPI_COMM_WORLD);

	if(r) {
		*r = 0.0;
		for(i = 0; i < n; i++) {
			*r += local_result[i];
		}
	}
	if(rs) {
		for(i = 0; i < n; i++) {
			rs[i] = local_result[i];
		}
	}
}

void VecNorm(void* x, void* layout, void* result, void* results) {
	VecDot(x, x, layout, result, results);
}

__END_DECLS__
