#include "Mesh.h"
#include "Field.h"
#include "csr.h"

#include "assembler.h"

__BEGIN_DECLS__

static __global__ void
AssemleTetFKernel(u32 num_node, const f64* xg,
									u32 num_elem, const u32* iel,
									f64* wgold, f64* dwgold, f64* dwg,
									f64* F) {
	
}
void assemble_system_tet(Mesh3D *mesh,
												 Field *wgold, Field* dwgold, Field* dwg,
												 f64* F, CSRMatrix* J) {

	__constant__ f64 shlu[16] = {0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105,
															 0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105,
															 0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105,
															 0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685};

	__constant__ f64 shlgradu[12] = {-1.0, -1.0, -1.0,
																	 1.0, 0.0, 0.0,
																	 0.0, 1.0, 0.0,
																	 0.0, 0.0, 1.0};
	const i32 NSHL = 4;
	/* Assume all 2D arraies are column-majored */
	for(u32 batch = 0; batch < num_batch; ++batch) {
		/* 0. Select the batch */
		/* 0.0. k = iel[:, batch], k.shape=(NSHL, lk) */
		/* 0.1. xg_batched[:, 0:NSHL, 0:lk] = xg[:, k] */
		/* 0.2. wgold_batched[:, ] = wgold[:, batch] */
		/* 0.3. dwgold_batched = dwgold[batch] */
		/* 0.4. dwg_batch = dwg[batch] */


		/* 1. Interpolate the field values */  
		/* 1.0. xq = xg_batch @ shlu */
		/* 1.1. wgold_q = wgold_batch @ shlu */
		/* 1.2. dwgold_q = dwgold_batch @ shlu */
		/* 1.3. dwg_q = dwg_batch @ shlu */

		/* 2. Calculate the elementwise dxi/dx */
		/* 2.0. dx/dxi = xq @ shlgradu */
		/* 2.1. dxi/dx = inv(dx/dxi) */
		/* 2.2. detJ = det(dx/dxi) */

		/* 3. Calculate the elementwise residual vector */
		/* */
	}	
}

__END_DECLS__
