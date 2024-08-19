#include <string.h>
#include <HYPRE.h>

int ffs(int);
#include <_hypre_utilities.h>
#include <_hypre_IJ_mv.h>

#include "alloc.h"
#include "Mesh.h"
#include "vec.h"

__BEGIN_DECLS__


void CDAM_VecCreate(MPI_Comm comm, void* mesh, CDAM_Vec** vec) {
	Mesh3D* mesh3d = (Mesh3D*)mesh;
	*vec = (CDAM_Vec*)CdamMallocHost(sizeof(CDAM_Vec));
	memset(*vec, 0, sizeof(CDAM_Vec));
	CDAM_VecComm(*vec) = comm;

	CDAM_VecPartitionLower(*vec) = CDAM_MeshLocalNodeBegin(mesh3d);
	CDAM_VecPartitionUpper(*vec) = CDAM_MeshLocalNodeEnd(mesh3d);
}

void CDAM_VecSetNumComponents(CDAM_Vec* vec, index_type num_components) {
	CDAM_VecNumComponents(vec) = num_components;
	CDAM_VecComponentSize(vec) = (index_type*)CdamMallocHost((num_components) * sizeof(index_type));
	memset(CDAM_VecComponentSize(vec), 0, (num_components) * sizeof(index_type));
	CDAM_VecSub(vec) = (HYPRE_IJVector*)CdamMallocHost(num_components * sizeof(HYPRE_IJVector));
	memset(CDAM_VecSub(vec), 0, num_components * sizeof(HYPRE_IJVector));
}

void CDAM_VecSetComponentSize(CDAM_Vec* vec, index_type component, index_type size) {
	CDAM_VecComponentSize(vec)[component] = size;
}

void CDAM_VecInitialize(CDAM_Vec* vec) {
	index_type i, bs;
	index_type num_components = CDAM_VecNumComponents(vec);
	index_type jlower = CDAM_VecPartitionLower(vec);
	index_type jupper = CDAM_VecPartitionUpper(vec);
	MPI_Comm comm = CDAM_VecComm(vec);
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
		offset[i + 1] = offset[i] + CDAM_VecComponentSize(vec)[i];
	}
	value_type* buffer = hypre_CTAlloc(value_type, offset[num_components], mem_location);

	for(i = 0; i < CDAM_VecNumComponents(vec); i++) {
		bs = CDAM_VecComponentSize(vec)[i];
		subvec = CDAM_VecSub(vec)[i];
		HYPRE_IJVectorCreate(comm, jlower * bs, jupper * bs, &subvec);
		HYPRE_IJVectorSetObjectType(subvec, HYPRE_PARCSR);
		HYPRE_IJVectorInitialize(subvec);
		ASSERT(hypre_IJVectorMemoryLocation(CDAM_VecSub(vec)[i]) == mem_location && "Memory location mismatch.\n");

		/* Free the old buffer and allocate a new one to ensure contiguous memory */
		hypre_TFree(CDAM_VecRawPtr(vec, i), mem_location);
		CDAM_VecRawPtr(vec, i) =  buffer + offset[i] * (jupper - jlower);
	}

	CdamFreeHost(offset, (num_components + 1) * sizeof(index_type));
}

void CDAM_VecDestroy(CDAM_Vec* vec) {
	index_type i;
	/* Retrieve the buffer pointer to avoid mem leak */
	value_type* buffer = CDAM_VecRawPtr(vec, 0);
	HYPRE_MemoryLocation mem_location;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
	mem_location = HYPRE_MEMORY_DEVICE;
#else
	mem_location = HYPRE_MEMORY_HOST;
#endif
	for(i = 0; i < CDAM_VecNumComponents(vec); i++) {
		/* Set the buffer pointer to NULL to avoid double-free */
		CDAM_VecRawPtr(vec, i) = NULL;
		HYPRE_IJVectorDestroy(CDAM_VecSub(vec)[i]);
	}
	CdamFreeHost(CDAM_VecComponentSize(vec), CDAM_VecNumComponents(vec) * sizeof(index_type));
	CdamFreeHost(CDAM_VecSub(vec), CDAM_VecNumComponents(vec) * sizeof(HYPRE_IJVector));
	CdamFreeHost(vec, sizeof(CDAM_Vec));
	hypre_TFree(buffer, mem_location); 
}

void CDAM_VecGetArray(CDAM_Vec* vec, value_type** array) {
	index_type i;
	*array = CDAM_VecRawPtr(vec, 0);
	for(i = 0; i < CDAM_VecNumComponents(vec); i++) {
		CDAM_VecRawPtr(vec, i) = NULL;
	}
}
void CDAM_VecRestoreArray(CDAM_Vec* vec, value_type** array) {
	index_type i, bs;
	index_type jlower = CDAM_VecPartitionLower(vec);
	index_type jupper = CDAM_VecPartitionUpper(vec);
	for(i = 0; i < CDAM_VecNumComponents(vec); i++) {
		bs = CDAM_VecComponentSize(vec)[i];
		CDAM_VecRawPtr(vec, i) = *array + bs * (jupper - jlower);
	}
	*array = NULL;
}

void CDAM_VecCopy(CDAM_Vec* src, CDAM_Vec* dest) {
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
	for(i = 0; i < CDAM_VecNumComponents(src); i++) {
		len += CDAM_VecComponentSize(src)[i];
	}
	len *= (CDAM_VecPartitionUpper(src) - CDAM_VecPartitionLower(src));

	CDAM_VecGetArray(src, &src_array);
	CDAM_VecGetArray(dest, &dest_array);

	hypre_TMemcpy(dest_array, src_array, value_type, len, mem_location, mem_location);

	CDAM_VecRestoreArray(src, &src_array);
	CDAM_VecRestoreArray(dest, &dest_array);
}

void CDAM_VecAssemble(CDAM_Vec* vec) {
	index_type i;
	for(i = 0; i < CDAM_VecNumComponents(vec); i++) {
		HYPRE_IJVectorAssemble(CDAM_VecSub(vec)[i]);
	}
}

void CDAM_VecAXPY(value_type alpha, CDAM_Vec* x, CDAM_Vec* y) {
	index_type i;
	for(i = 0; i < CDAM_VecNumComponents(x); i++) {
		hypre_SeqVectorAxpy(alpha, CDAM_VecSubSeqVec(x, i), CDAM_VecSubSeqVec(y, i));
	}
}

void CDAM_VecScale(CDAM_Vec* x, value_type alpha) {
	index_type i;
	for(i = 0; i < CDAM_VecNumComponents(x); i++) {
		hypre_SeqVectorScale(alpha, CDAM_VecSubSeqVec(x, i));
	}
}

void CDAM_VecZero(CDAM_Vec* x) {
	index_type i;
	for(i = 0; i < CDAM_VecNumComponents(x); i++) {
		hypre_SeqVectorSetConstantValues(CDAM_VecSubSeqVec(x, i), (value_type)0.0);
	}
}

void CDAM_VecInnerProd(CDAM_Vec* x, CDAM_Vec* y, value_type* result) {
	index_type i;
	value_type local_result = 0.0;
	for(i = 0; i < CDAM_VecNumComponents(x); i++) {
		local_result += hypre_SeqVectorInnerProd(CDAM_VecSubSeqVec(x, i), CDAM_VecSubSeqVec(y, i));
	}
	hypre_MPI_Allreduce(&local_result, result, 1, HYPRE_MPI_REAL,
                       hypre_MPI_SUM, CDAM_VecComm(x));
}

void CDAM_VecNorm(CDAM_Vec* x, value_type* result) {
	CDAM_VecInnerProd(x, x, result);
	*result = sqrt(*result);
}

void CDAM_VecView(CDAM_Vec* x, FILE* ostream) {}

__END_DECLS__
