#ifndef __H5UTIL_H__
#define __H5UTIL_H__

#include <hdf5.h>
#include "common.h"


// Function prototypes
__BEGIN_DECLS__


typedef struct H5FileStatus H5FileStatus;
struct H5FileStatus {
	hid_t file_id;
};

/* read h5 file */
/* Check the existence of the dataset */
bool H5CheckDataset(hid_t file_id, const char *dataset_name);

/* Get the size of the dataset */
void H5GetDatasetSize(hid_t file_id, const char *dataset_name, hsize_t *dims);


__END_DECLS__

#endif // __H5UTIL_H__
