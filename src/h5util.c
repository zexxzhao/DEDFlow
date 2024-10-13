#include "alloc.h"
#include "h5util.h"

__BEGIN_DECLS__

/* Open an HDF5 file */
H5FileInfo* H5OpenFile(const char* filename, const char* mode) {
	H5FileInfo* h5file = CdamTMalloc(H5FileInfo, 1, HOST_MEM);

	if (strcmp(mode, "r") == 0) {
		h5file->file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	}
	else if (strcmp(mode, "w") == 0) {
		h5file->file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	}
	else if (strcmp(mode, "a") == 0) {
		h5file->file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
	}
	else {
		ASSERT(false && "H5OpenFile: Invalid mode!");
	}
	ASSERT(h5file->file_id != H5I_INVALID_HID && "H5OpenFile: Failed to open file!");

	memcpy(h5file->filename, filename, strlen(filename));
	return h5file;
}

/* Close an HDF5 file */
void H5CloseFile(H5FileInfo* h5file) {
	H5Fclose(h5file->file_id);
	CdamFree(h5file, sizeof(H5FileInfo), HOST_MEM);
}

/* Check the existence of the file */
b32 H5FileExist(const char* filename) {
	hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id < 0) {
		return false;
	}
	H5Fclose(file_id);
	return true;
}
/* Check the file mode */
b32 H5FileIsWritable(H5FileInfo* h5file) {
	unsigned intent;
	hid_t file_id = h5file->file_id;
	H5Fget_intent(file_id, &intent);
	return intent & (H5F_ACC_RDWR | H5F_ACC_TRUNC);
}

b32 H5FileIsReadable(H5FileInfo* h5file) {
	unsigned intent;
	hid_t file_id = h5file->file_id;
	H5Fget_intent(file_id, &intent);
	return intent == H5F_ACC_RDONLY || intent & H5F_ACC_RDWR;
}

/* Check the existence of a group */
b32 H5GroupExist(H5FileInfo* h5file, const char* group_name) {
	hid_t file_id = h5file->file_id;
	herr_t status;
	H5O_info_t obj_info;
	if(! H5Lexists(file_id, group_name, H5P_DEFAULT)) {
		return false;
	}
#if H5_VERSION_GE(1, 12, 0)
	status = H5Oget_info_by_name(file_id, group_name, &obj_info, H5P_DEFAULT, H5P_DEFAULT);
#else
	status = H5Oget_info_by_name(file_id, group_name, &obj_info, H5P_DEFAULT);
#endif
	return status >= 0 && obj_info.type == H5O_TYPE_GROUP;
}
/* Check the existence of a dataset */
b32 H5DatasetExist(H5FileInfo* h5file, const char* dataset_name) {
	hid_t file_id = h5file->file_id;
	herr_t status;
	H5O_info_t obj_info;
	if(! H5Lexists(file_id, dataset_name, H5P_DEFAULT)) {
		return false;
	}
#if H5_VERSION_GE(1, 12, 0)
	status = H5Oget_info_by_name(file_id, dataset_name, &obj_info, H5P_DEFAULT, H5P_DEFAULT);
#else
	status = H5Oget_info_by_name(file_id, dataset_name, &obj_info, H5P_DEFAULT);
#endif
	return status >= 0 && obj_info.type == H5O_TYPE_DATASET;
}

void H5GetDatasetSize(H5FileInfo* h5file, const char* dataset_name, index_type* size) {
	if(!H5DatasetExist(h5file, dataset_name)) {
		*size = 0;
		return;
	}
	hid_t dataset_id = H5Dopen(h5file->file_id, dataset_name, H5P_DEFAULT);
	hid_t dataspace_id = H5Dget_space(dataset_id);
	hsize_t dataset_size;
	i32 ndims;

	ndims = H5Sget_simple_extent_ndims(dataspace_id);
	ASSERT(ndims == 1 && "All arraies are flattened into 1D.\n");

	H5Sget_simple_extent_dims(dataspace_id, &dataset_size, NULL);
	ASSERT(dataset_size < UINT_MAX && "H5GetDatasetSize: Dataset size exceeds the limit of index_type.\n");
	*size = (index_type)dataset_size;

	H5Sclose(dataspace_id);
	H5Dclose(dataset_id);
}


static void
H5ReadDataset(H5FileInfo* h5file, const char* dataset_name,
							hid_t mem_type_id, void* data) {
	hid_t dataset_id = H5Dopen(h5file->file_id, dataset_name, H5P_DEFAULT);
	// hid_t dataspace_id = H5Dget_space(dataset_id);
	// i32 ndims;
	// hsize_t dataset_size;

	// ndims = H5Sget_simple_extent_ndims(dataspace_id);
	// ASSERT(ndims == 1 && "All arraies are flattened into 1D.\n");

	// hsize_t* h5dims = (hsize_t*)malloc(ndims * sizeof(hsize_t));
	// H5Sget_simple_extent_dims(dataspace_id, &dataset_size, NULL);
	// hsize_t size = 1;
	// for (i32 i = 0; i < ndims; i++) {
	// 	size *= h5dims[i];
	// }
	// free(h5dims);
	H5Dread(dataset_id, mem_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	// H5Sclose(dataspace_id);
	H5Dclose(dataset_id);
}

void H5ReadDataseti32(H5FileInfo* h5file, const char* dataset_name, i32* data) {
	H5ReadDataset(h5file, dataset_name, H5T_NATIVE_INT32, (void*)data);
}

void H5ReadDatasetu32(H5FileInfo* h5file, const char* dataset_name, u32* data) {
	H5ReadDataset(h5file, dataset_name, H5T_NATIVE_UINT32, (void*)data);
}

void H5ReadDatasetf32(H5FileInfo* h5file, const char* dataset_name, f32* data) {
	H5ReadDataset(h5file, dataset_name, H5T_NATIVE_FLOAT, (void*)data);
}

void H5ReadDatasetf64(H5FileInfo* h5file, const char* dataset_name, f64* data) {
	H5ReadDataset(h5file, dataset_name, H5T_NATIVE_DOUBLE, (void*)data);
}


void H5ReadDatasetInd(H5FileInfo* h5file, const char* dataset_name, index_type* data) {
#if defined(USE_U32_INDEX)
	H5ReadDatasetu32(h5file, dataset_name, data);
#elif defined(USE_I32_INDEX)
	H5ReadDataseti32(h5file, dataset_name, data);
#elif defined(USE_U64_INDEX)
#error "H5ReadDatasetInd: U64 index is not supported yet!"
#elif defined(USE_I64_INDEX)
#error "H5ReadDatasetInd: I64 index is not supported yet!"
#else
#error "H5ReadDatasetInd: No index type is defined!"
#endif
}

void H5ReadDatasetVal(H5FileInfo* h5file, const char* dataset_name, value_type* data) {
#if defined(USE_F32_VALUE)
	H5ReadDatasetf32(h5file, dataset_name, data);
#elif defined(USE_F64_VALUE)
	H5ReadDatasetf64(h5file, dataset_name, data);
#else
#error "H5ReadDatasetVal: No value type is defined!"
#endif
}


void H5ReadDatasetValIndexed(H5FileInfo* h5file, const char* dataset_name,
														 index_type size, index_type* index,
														 value_type* data) {
	hid_t dataset_id = H5Dopen(h5file->file_id, dataset_name, H5P_DEFAULT);
	hid_t dataspace_id = H5Dget_space(dataset_id);
	hid_t mem_type_id;
#if defined(USE_F32_VALUE)
	mem_type_id = H5T_NATIVE_FLOAT;
#elif defined(USE_F64_VALUE)
	mem_type_id = H5T_NATIVE_DOUBLE;
#else
#error "H5ReadDatasetValIndexed: No value type is defined!"
#endif

	hsize_t h5len = size;
	hid_t memspace_id = H5Screate_simple(1, &h5len, NULL);

	hsize_t* hindex = CdamTMalloc(hsize_t, size, HOST_MEM);
	for (index_type i = 0; i < size; i++) {
		hindex[i] = index[i];
	}

	H5Sselect_elements(dataspace_id, H5S_SELECT_SET, size, (const hsize_t*)hindex);


	H5Dread(dataset_id, mem_type_id, memspace_id, dataspace_id, H5P_DEFAULT, data);


	CdamFree(hindex, size * sizeof(hsize_t), HOST_MEM);
	H5Sclose(memspace_id);
	H5Sclose(dataspace_id);
	H5Dclose(dataset_id);

}


/* Write a dataset to an HDF5 file */
static void
H5WriteDataset(H5FileInfo* h5file, const char *dataset_name,
							 hid_t mem_type_id, index_type len, const void* data) {
	hid_t group_id;
	i32 index = 0;
	char buff[256];
	hsize_t h5len = len;
	hid_t dataspace_id = H5Screate_simple(1, &h5len, NULL);
	ASSERT(dataspace_id >= 0 && "H5WriteDataset: Failed to create dataspace!");
	memset(buff, 0, 256);
	while(dataset_name[index] != '\0') {
		if (dataset_name[index] == '/') {
			strncpy(buff, dataset_name, index);
			if (H5GroupExist(h5file, buff)) {
				group_id = H5Gopen(h5file->file_id, buff, H5P_DEFAULT);
				/* printf("H5WriteDataset: Opened group %s\n", buff); */
			}
			else {
				group_id = H5Gcreate(h5file->file_id, buff, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
				/* printf("H5WriteDataset: Created group %s\n", buff); */
			}
			ASSERT(group_id >= 0 && "H5WriteDataset: Failed to create group!");
			H5Gclose(group_id);
		}
		index++;
	}
	hid_t dataset_id = H5Dcreate(h5file->file_id, dataset_name,
															 mem_type_id, dataspace_id,
															 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	ASSERT(dataset_id >= 0 && "H5WriteDataset: Failed to create dataset!");
	H5Dwrite(dataset_id, mem_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	H5Dclose(dataset_id);
	H5Sclose(dataspace_id);
}


void H5WriteDataseti32(H5FileInfo* h5file, const char *dataset_name, index_type len, const i32* data) {
	H5WriteDataset(h5file, dataset_name, H5T_NATIVE_INT32, len, (const void*)data);
}

void H5WriteDatasetu32(H5FileInfo* h5file, const char *dataset_name, index_type len, const u32* data) {
	H5WriteDataset(h5file, dataset_name, H5T_NATIVE_UINT32, len, (const void*)data);
}

void H5WriteDatasetf32(H5FileInfo* h5file, const char *dataset_name, index_type len, const f32* data) {
	H5WriteDataset(h5file, dataset_name, H5T_NATIVE_FLOAT, len, (const void*)data);
}

void H5WriteDatasetf64(H5FileInfo* h5file, const char *dataset_name, index_type len, const f64* data) {
	H5WriteDataset(h5file, dataset_name, H5T_NATIVE_DOUBLE, len, (const void*)data);
}

void H5WriteDatasetInd(H5FileInfo* h5file, const char *dataset_name, index_type len, const index_type* data) {
#if defined(USE_U32_INDEX)
	H5WriteDatasetu32(h5file, dataset_name, len, data);
#elif defined(USE_I32_INDEX)
	H5WriteDataseti32(h5file, dataset_name, len, data);
#elif defined(USE_U64_INDEX)
#error "H5WriteDatasetInd: U64 index is not supported yet!"
#elif defined(USE_I64_INDEX)
#error "H5WriteDatasetInd: I64 index is not supported yet!"
#else
#error "H5WriteDatasetInd: No index type is defined!"
#endif
}

void H5WriteDatasetVal(H5FileInfo* h5file, const char *dataset_name, index_type len, const value_type* data) {
#if defined(USE_F32_VALUE)
	H5WriteDatasetf32(h5file, dataset_name, len, (const f32*)data);
#elif defined(USE_F64_VALUE)
	H5WriteDatasetf64(h5file, dataset_name, len, (const f64*)data);
#else
#error "H5WriteDatasetVal: No value type is defined!"
#endif
}
__END_DECLS__
