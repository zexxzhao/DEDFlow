#ifndef __CSR_H__
#define __CSR_H__

#include "common.h"
#include "blas.h"

__BEGIN_DECLS__


typedef struct CSRAttr CSRAttr;
struct CSRAttr {
	index_type num_row;
	index_type num_col;
	index_type nnz;
	index_type *row_ptr;
	index_type *col_ind;
	const CSRAttr *parent;
};

#define CSRAttrNumRow(attr) ((attr)->num_row)
#define CSRAttrNumCol(attr) ((attr)->num_col)
#define CSRAttrNNZ(attr) ((attr)->nnz)
#define CSRAttrRowPtr(attr) ((attr)->row_ptr)
#define CSRAttrColInd(attr) ((attr)->col_ind)

CSRAttr* CSRAttrCreate(void* pmesh);
void CSRAttrDestroy(CSRAttr *attr);

CSRAttr* CSRAttrCreateBlock(const CSRAttr* attr, index_type block_row, index_type block_col);

index_type CSRAttrLength(CSRAttr *attr, index_type row);
index_type* CSRAttrRow(CSRAttr *attr, index_type row);

void CSRAttrGetNonzeroIndBatched(const CSRAttr* attr, index_type batch_size, const index_type* row, const index_type* col, index_type* ind);


__END_DECLS__

#endif /* __CSR_H__ */
