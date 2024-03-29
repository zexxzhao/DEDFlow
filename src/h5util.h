#ifndef __H5UTIL_H__
#define __H5UTIL_H__

#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include "common.h"


// Function prototypes
__BEGIN_DECLS__


typedef struct H5FileInfo H5FileInfo;
struct H5FileInfo {
	char filename[256];
	hid_t file_id;
};

/* Open an HDF5 file */
H5FileInfo* H5OpenFile(const char* filename, const char* mode);
/* Close an HDF5 file */
void H5CloseFile(H5FileInfo* h5file);

/* Check the existence of the file */
b32 H5FileExist(const char* filename);

/* Check the file mode */
b32 H5FileIsWritable(H5FileInfo* h5file);
b32 H5FileIsReadable(H5FileInfo* h5file);

/* Read h5 file */
/* Check the existence of the group */
b32 H5GroupExist(H5FileInfo* h5file, const char *group_name);
/* Check the existence of the dataset */
b32 H5DatasetExist(H5FileInfo* h5file, const char *dataset_name);

/* Get the size of the dataset */
void H5GetDatasetSize(H5FileInfo* h5file, const char* dataset_name, u32* size);

/* Read the dataset */
void H5ReadDataseti32(H5FileInfo* h5file, const char *dataset_name, i32* data);
void H5ReadDatasetu32(H5FileInfo* h5file, const char *dataset_name, u32* data);
void H5ReadDatasetf32(H5FileInfo* h5file, const char *dataset_name, f32* data);
void H5ReadDatasetf64(H5FileInfo* h5file, const char *dataset_name, f64* data);

/* Write h5 file */
void H5WriteDataseti32(H5FileInfo* h5file, const char *dataset_name, u32 len, const i32* data);
void H5WriteDatasetu32(H5FileInfo* h5file, const char *dataset_name, u32 len, const u32* data);
void H5WriteDatasetf32(H5FileInfo* h5file, const char *dataset_name, u32 len, const f32* data);
void H5WriteDatasetf64(H5FileInfo* h5file, const char *dataset_name, u32 len, const f64* data);

__END_DECLS__

#endif // __H5UTIL_H__
