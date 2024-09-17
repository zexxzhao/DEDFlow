#ifndef __CSR_H__
#define __CSR_H__

#include <cusparse.h>
#include "common.h"

__BEGIN_DECLS__

typedef struct CdamMesh CdamMesh;

typedef index_type csr_index_type;
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

CSRAttr* CSRAttrCreate(CdamMesh* mesh);
void CSRAttrDestroy(CSRAttr *attr);

CSRAttr* CSRAttrCreateBlock(const CSRAttr* attr, csr_index_type block_row, csr_index_type block_col);

index_type CSRAttrLength(CSRAttr *attr, csr_index_type row);
csr_index_type* CSRAttrRow(CSRAttr *attr, csr_index_type row);

void CSRAttrGetNonzeroIndBatched(const CSRAttr* attr, csr_index_type batch_size, const index_type* row, const index_type* col, index_type* ind);


__END_DECLS__

#endif /* __CSR_H__ */
