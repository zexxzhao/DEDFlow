#ifndef __CSR_H__
#define __CSR_H__

#include <cusparse.h>
#include "common.h"

__BEGIN_DECLS__

typedef struct Mesh3D Mesh3D;

typedef u32 csr_index_type;
typedef struct CSRAttr CSRAttr;
struct CSRAttr {
	csr_index_type num_row;
	csr_index_type num_col;
	csr_index_type nnz;
	csr_index_type *row_ptr;
	csr_index_type *col_ind;
};

#define CSRAttrNumRow(attr) ((attr)->num_row)
#define CSRAttrNumCol(attr) ((attr)->num_col)
#define CSRAttrNNZ(attr) ((attr)->nnz)
#define CSRAttrRowPtr(attr) ((attr)->row_ptr)
#define CSRAttrColInd(attr) ((attr)->col_ind)

CSRAttr* CSRAttrCreate(const Mesh3D* mesh, csr_index_type block_size);
void CSRAttrDestroy(CSRAttr *attr);

u32 CSRAttrLength(CSRAttr *attr, csr_index_type row);
csr_index_type* CSRAttrRow(CSRAttr *attr, csr_index_type row);


typedef struct CSRMatrix CSRMatrix;
typedef f64 csr_value_type;
struct CSRMatrix {
	b32 external_attr;
	CSRAttr *attr;
	csr_value_type *data;
	cusparseSpMatDescr_t descr;
};

#define CSRMatrixAttr(matrix) ((matrix)->attr)
#define CSRMatrixRowPtr(matrix) (CSRAttrRowPtr(CSRMatrixAttr(matrix)))
#define CSRMatrixColInd(matrix) (CSRAttrColInd(CSRMatrixAttr(matrix)))
#define CSRMatrixData(matrix) ((matrix)->data)
#define CSRMatrixNumRow(matrix) (CSRAttrNumRow(CSRMatrixAttr(matrix)))
#define CSRMatrixNumCol(matrix) (CSRAttrNumCol(CSRMatrixAttr(matrix)))
#define CSRMatrixNNZ(matrix) (CSRAttrNNZ(CSRMatrixAttr(matrix)))
#define CSRMatrixDescr(matrix) ((matrix)->descr)

CSRMatrix* CSRMatrixCreate(CSRAttr *attr);
CSRMatrix* CSRMatrixCreateMesh(const Mesh3D* mesh, csr_index_type block_size);

void CSRMatrixDestroy(CSRMatrix *matrix);



__END_DECLS__

#endif /* __CSR_H__ */
