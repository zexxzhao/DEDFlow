#include "Mesh.h"
#include "Field.h"
#include "csr.h"

#include "assemble.h"

#define kRHOC (0.5)
#define kDT (0.1)
#define kALPHAM ((3.0 - kRHOC) / (1.0 + kRHOC))
#define kALPHAF (1.0 / (1.0 + kRHOC))
#define kGAMMA (0.5 + kALPHAM - kALPHAF)

__BEGIN_DECLS__
#define NQR 4
__constant__ f64 gw[4] = {0.0416666666666667, 0.0416666666666667, 0.0416666666666667, 0.0416666666666667};
__constant__ f64 shlu[16] = {0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105,
														 0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105,
														 0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105,
														 0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685};

__constant__ f64 shlgradu[12] = {-1.0, -1.0, -1.0,
																 1.0, 0.0, 0.0,
																 0.0, 1.0, 0.0,
																 0.0, 0.0, 1.0};

/**
* @brief Load the values from the source array to the destination array
* @param[in]  count The number of values to be loaded
* @param[in]  index The index of the values to be loaded, whose size is (count*inc,)
* @param[in]  inc   The increment of the index
* @param[in]  bs    The length of the block
* @param[in]  src   The source array to be loaded. The array is column-majored of size (block_length, *)
* @param[out] dst   The destination array to be loaded. The array is column-majored of size (block_length, count)
*/
static __global__
template<typename Index, typename Real> void
LoadValueKernel(Index count, const Index* index, Index inc, Index block_length, const Real* src, Real* dst) {

	i32 idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if(idx >= count) return;

	const u32* index_ptr = index + idx * inc;
	f64* dst_ptr = dst + idx * block_length;
	while(block_length >= 4) {
		*((double4*)dst_ptr) = *((double4*)(src + index_ptr[0]));
		block_length -= 4;
		dst_ptr += 4;
		index_ptr += 4;
	}
	if(block_length >= 2) {
		*((double2*)dst_ptr) = *((double2*)(src + index_ptr[0]));
		block_length -= 2;
		dst_ptr += 2;
		index_ptr += 2;
	}
	if(block_length >= 1) {
		*dst_ptr = src[index_ptr[0]];
	}

}

/**
 * @brief Load the values from the source array and apply axpy operation to the destination array
 */
static __global__ void
LoadValueAxpyKernel(u32 count, const u32* index, u32 inc, u32 block_length, const f64* src, f64* dst, f64 alpha) {
	i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= count) return;

	const u32* index_ptr = index + idx * inc;
	f64* dst_ptr = dst + idx * block_length;
	while(block_length >= 4) {
		*((double4*)dst_ptr) = fma(*((double4*)(src + index_ptr[0])), alpha, *((double4*)dst_ptr));
		block_length -= 4;
		dst_ptr += 4;
		index_ptr += 4;
	}
	if(block_length >= 2) {
		*((double2*)dst_ptr) = fma(*((double2*)(src + index_ptr[0])), alpha, *((double2*)dst_ptr));
		block_length -= 2;
		dst_ptr += 2;
		index_ptr += 2;
	}
	if(block_length >= 1) {
		*dst_ptr = fma(src[index_ptr[0]], alpha, *dst_ptr);
	}
}

/**
 * @brief Calculate the inverse of the Jacobian matrix of the element
 * @param[in] num_elem The number of elements
 * @param[in] iel      The index of the element in the mesh
 * @param[in] xg       The global coordinates of the nodes
 * @param[out] elem_invJ The inverse of the Jacobian matrix of the element
 */
static __global__ void
GetElemInvJ3DKernel(u32 num_elem, const u32* iel, const f64* xg, f64* elem_invJ) {
	i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num_elem) return;

	const f64* xg_ptr0 = xg + iel[idx*NSHL+0] * 3;
	const f64* xg_ptr1 = xg + iel[idx*NSHL+1] * 3;
	const f64* xg_ptr2 = xg + iel[idx*NSHL+2] * 3;
	const f64* xg_ptr3 = xg + iel[idx*NSHL+3] * 3;
	f64* elem_invJ_ptr = elem_invJ + idx * 10;
	f64 dxidx[9];
	dxidx[0] = xg_ptr1[0] - xg_ptr0[0];
	dxidx[1] = xg_ptr1[1] - xg_ptr0[1];
	dxidx[2] = xg_ptr1[2] - xg_ptr0[2];
	dxidx[3] = xg_ptr2[0] - xg_ptr0[0];
	dxidx[4] = xg_ptr2[1] - xg_ptr0[1];
	dxidx[5] = xg_ptr2[2] - xg_ptr0[2];
	dxidx[6] = xg_ptr3[0] - xg_ptr0[0];
	dxidx[7] = xg_ptr3[1] - xg_ptr0[1];
	dxidx[8] = xg_ptr3[2] - xg_ptr0[2];

	f64 detJ = dxidx[0] * (dxidx[4] * dxidx[8] - dxidx[5] * dxidx[7]) -
						 dxidx[1] * (dxidx[3] * dxidx[8] - dxidx[5] * dxidx[6]) +
						 dxidx[2] * (dxidx[3] * dxidx[7] - dxidx[4] * dxidx[6]);
	elem_invJ_ptr[0] = (dxidx[4] * dxidx[8] - dxidx[5] * dxidx[7]) / detJ;
	elem_invJ_ptr[1] = (dxidx[2] * dxidx[7] - dxidx[1] * dxidx[8]) / detJ;
	elem_invJ_ptr[2] = (dxidx[1] * dxidx[5] - dxidx[2] * dxidx[4]) / detJ;
	elem_invJ_ptr[3] = (dxidx[5] * dxidx[6] - dxidx[3] * dxidx[8]) / detJ;
	elem_invJ_ptr[4] = (dxidx[0] * dxidx[8] - dxidx[2] * dxidx[6]) / detJ;
	elem_invJ_ptr[5] = (dxidx[2] * dxidx[3] - dxidx[0] * dxidx[5]) / detJ;
	elem_invJ_ptr[6] = (dxidx[3] * dxidx[7] - dxidx[4] * dxidx[6]) / detJ;
	elem_invJ_ptr[7] = (dxidx[1] * dxidx[6] - dxidx[0] * dxidx[7]) / detJ;
	elem_invJ_ptr[8] = (dxidx[0] * dxidx[4] - dxidx[1] * dxidx[3]) / detJ;
	elem_invJ_ptr[9] = detJ;
}

static __global__ void
GetShapeGradKernel(u32 num_elem, const f64* elem_invJ, f64* shgradl) {
	i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num_elem) return;

	const f64* elem_invJ_ptr = elem_invJ + idx * 10;
	f64* shgradl_ptr = shgradl + idx * 12;
	f64 dxidx[9] = {elem_invJ_ptr[0], elem_invJ_ptr[1], elem_invJ_ptr[2],
									elem_invJ_ptr[3], elem_invJ_ptr[4], elem_invJ_ptr[5],
									elem_invJ_ptr[6], elem_invJ_ptr[7], elem_invJ_ptr[8]};

	#pragma unroll
	for(u32 s = 0; s < 4; ++s) {
		f64 shgradu[3] = {shlgradu[s*3+0], shlgradu[s*3+1], shlgradu[s*3+2]};
		shgradl_ptr[s*3+0] = shgradu[0] * dxidx[0] + shgradu[1] * dxidx[3] + shgradu[2] * dxidx[6];
		shgradl_ptr[s*3+1] = shgradu[0] * dxidx[1] + shgradu[1] * dxidx[4] + shgradu[2] * dxidx[7];
		shgradl_ptr[s*3+2] = shgradu[0] * dxidx[2] + shgradu[1] * dxidx[5] + shgradu[2] * dxidx[8];
	}
}
static __global__ void
AssemleTetWeakFormKernel(u32 num_elem, const u32* iel, const f64* elem_invJ,
												 const f64* qr_wgalpha, const f64* qr_dwgalpha, const f64* qr_wggradalpha,
												 const f64* shgradl,
												 f64* elem_F, f64* elem_J) {
	i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= num_elem) return;

	const f64* elem_invJ_ptr = elem_invJ + idx * 10;
	const f64 detJ = elem_invJ_ptr[9];
	const f64* qr_wgalpha_ptr = qr_wgalpha + idx * 4;
	const f64* qr_dwgalpha_ptr = qr_dwgalpha + idx * 4;
	const f64* qr_wggradalpha_ptr = qr_wggradalpha + idx * 3;
	const f64* shgradl_ptr = shgradl + idx * 12;

	f64* elem_F_ptr = elem_F + idx * 4;
	f64* elem_J_ptr = elem_J + idx * 16;

	f64 F_ptr[4+16] = {0.0, 0.0, 0.0, 0.0,
								 0.0, 0.0, 0.0, 0.0,
								 0.0, 0.0, 0.0, 0.0,
								 0.0, 0.0, 0.0, 0.0,
								 0.0, 0.0, 0.0, 0.0};
	f64* J_ptr = F_ptr + 4;
	
	/* Weak form: */
	/* F_A[0:NSHL] += (gw[0:NQR] * detJ * qr_dwgalpha_ptr[0:NQR]) @ shl[0:NQR, 0:NSHL] */
	/* F_A[0:NSHL] += sum(gw[0:NQR]) * detJ * qr_wggradalpha_ptr[0:3] @ shgradl[0:3, 0:NSHL] */
	/* J_AB[0:NSHL, 0:NSHL] += kALPHAM * detJ * shl[0,NQR, NSHL].T @ (gw[0:NQR, None] * shl[0:NQR, 0:NSHL]) */
	/* J_AB[0:NSHL, 0:NSHL] += kDT * kALPHAF * kGAMMA * sum(gw[0:NQR]) * detJ 
													 * shgradl[0:3, 0:NSHL].T @ shgradl[0:3, 0:NSHL] */

	if(elem_F) {
		#pragma unroll
		for(u32 q = 0; q < NQR; ++q) {
			f64 qr_dwgalpha_q = qr_dwgalpha_ptr[q];
			f64 qr_wggradalpha_q = qr_wggradalpha_ptr[q];
			#pragma unroll
			for(u32 s = 0; s < NSHL; ++s) {
				F_ptr[s] += gw[q] * detJ * qr_dwgalpha_q * shgradl_ptr[q*3+s];
				F_ptr[s] += gw[q] * detJ * qr_wggradalpha_q * shgradl_ptr[q*3+s];
			}
		}

		#pragma unroll
		for(u32 s = 0; s < NSHL; ++s) {
			#pragma unroll
			for(u32 t = 0; t < NSHL; ++t) {
				elem_F_ptr[s] += F_ptr[s*NSHL+t];
			}
		}
	}
	if(elem_J) {
		#pragma unroll
		for(u32 q = 0; q < NQR; ++q) {
			#pragma unroll
			for(u32 s = 0; s < NSHL; ++s) {
				#pragma unroll
				for(u32 t = 0; t < NSHL; ++t) {
					J_ptr[s*NSHL+t] += kALPHAM * detJ * shlgradu[q*3+s] * gw[q] * shlgradu[q*3+t];
					J_ptr[s*NSHL+t] += kDT * kALPHAF * kGAMMA * detJ * shgradl_ptr[q*3+s] * shgradl_ptr[q*3+t];
				}
			}
		}

		#pragma unroll
		for(u32 s = 0; s < NSHL; ++s) {
			#pragma unroll
			for(u32 t = 0; t < NSHL; ++t) {
				elem_J_ptr[s*NSHL+t] = J_ptr[s*NSHL+t];
			}
		}
	
	}
}

struct AssembleGraph {
	u32 num_elem;

};

void AssembleSystemTet(Mesh3D *mesh,
											 Field *wgold, Field* dwgold, Field* dwg,
											 f64* F, CSRMatrix* J) {

	const i32 NSHL = 4;
	const u32 num_tet = Mesh3DNumNode(mesh);
	const u32* iel_tet = Mesh3DTet(mesh);
	const f64* xg = Mesh3DCoord(mesh);

	f64* wgold_dptr = ArrayData(FieldDevice(wgold));
	f64* dwgold_dptr = ArrayData(FieldDevice(dwgold));
	f64* dwg_dptr = ArrayData(FieldDevice(dwg));


	f64* buffer = (f64*)CdamMallocDevice(num_tet * sizeof(f64));
	f64* elem_invJ = (f64*)CdamMallocDevice(num_tet * sizeof(f64) * 10);
	f64* shgradl = (f64*)CdamMallocDevice(num_tet * sizeof(f64) * NSHL * 3);

	f64* elem_F = (f64*)CdamMallocDevice(num_tet * sizeof(f64) * NSHL);
	f64* elem_J = (f64*)CdamMallocDevice(num_tet * sizeof(f64) * NSHL * NSHL);

	f64* qr_wgalpha = (f64*)CdamMallocDevice(2 * num_tet * sizeof(f64) * NQR);
	f64* qr_dwgalpha = qr_wgalpha + num_tet * 4 * sizeof(f64);
	f64* qr_wggradalpha = (f64*)CdamMallocDevice(2 * num_tet * sizeof(f64) * 3);

	i32 num_thread = 256;
	i32 num_block = (num_tet + num_thread - 1) / num_thread;
	cublasHandle_t handle;
	const f64 one = 1.0, zero = 0.0, minus_one = -1.0;

	cublasCreate(&handle);

	/* Assume all 2D arraies are column-majored */
	for(u32 b = 0; b < num_batch; ++b) {
		/* 0. Calculate the element metrics */
		/* 0.0. Get dxi/dx and det(dxi/dx) */
		GetElemInvJ3DKernel<<<num_block, num_thread>>>(num_tet, iel_tet, xg, elem_invJ);
		/* 0.1. Calculate the gradient of the shape functions: 
						shgradl[d, s, :] = \sum_d0 shlgradu[d0, s] * elem_invJ[d0, d, :] */
		GetShapeGradKernel<<<num_block, num_thread>>>(num_tet, elem_invJ, shgradl);

		/* 1. Interpolate the field values */  
		/* 1.0. Calculate qr_wgalpha */
		cudaMemset(qr_wgalpha, 0, 2 * num_tet * sizeof(f64) * NQR);
		cudaMemset(qr_wggradalpha, 0, 2 * num_tet * sizeof(f64) * 3);
		for(u32 v = 0; v < NSHL; ++v) {
			/* 1.0.0. buffer[:] = wgold[iel_tet[v, :]] */
			LoadValueKernel<<<num_block, num_thread>>>(num_tet, iel_tet, NSHL, 1, wgold_dptr, buffer);
			/* 1.0.1. buffer[:] += kDT * kALPHAF * (1.0 - kGAMMA) * dwgold[iel_tet[v, :]] */ 
			LoadValueAXPYKernel<<<num_block, num_thread>>>(num_tet, iel_tet, NSHL, 1, dwgold_dptr, buffer, kDT * kALPHAF * (1.0 - kGAMMA)); 
			/* 1.0.2. buffer[:] += kDT * kALPHAF * kGAMMA * dwg[iel_tet[v, :]] */
			LoadValueAXPYKernel<<<num_block, num_thread>>>(num_tet, iel_tet, NSHL, 1, dwg_dptr, buffer, kDT * kALPHAF * kGAMMA);
			/* 1.0.3. qr_wgalpha[0:NQR, :] += shlu[0:NQR, v] * buffer[:] */
			cublasDger(handle, NQR, num_tet, 1.0, shlu + v * NQR, NQR, buffer, 1, qr_wgalpha, NQR);
		}
		/* 1.1. Calculate qr_dwgalpha */
		cudaMemset(qr_dwgalpha, 0, num_tet * sizeof(f64) * NQR);
		cudaMemset(qr_wggradalpha, 0, 2 * num_tet * sizeof(f64) * 3);
		for(u32 v = 0; v < NSHL; ++v) {
			/* 1.1.0. buffer[:] = (1.0 - kALPHAM) * dwgold[iel_tet[v, :]] */
			LoadValueAXPYKernel<<<num_block, num_thread>>>(num_tet, iel_tet, NSHL, 1, dwgold_dptr, buffer, 1.0 - kALPHAM);
			/* 1.1.1. buffer[:] = kALPHAM * dwg[iel_tet[v, :]] */
			LoadValueAXPYKernel<<<num_block, num_thread>>>(num_tet, iel_tet, NSHL, 1, dwg_dptr, buffer, kALPHAM);
			/* 1.1.2. qr_dwgalpha[0:NQR, :] += shlu[0:NQR, v] * buffer[:] */
			cublasDger(handle, NQR, num_tet, 1.0, shlu + v * NQR, NQR, buffer, 1, qr_dwgalpha, NQR);
		}


		/* 2. Calculate the elementwise residual vector and matrix */
		AssemleTetWeakFormKernel<<<num_block, num_thread>>>(num_tet, iel_tet, elem_invJ,
																													qr_wgalpha, qr_dwgalpha, qr_wggradalpha,
																													shgradl,
																													elem_F, elem_J);
		/* 3. Assemble the global residual vector and matrix */
		if(F) {
			/* F[:] += elem_F[:] */
			cublasDaxpy(handle, num_tet * NSHL, one, elem_F, 1, F, 1);
		}
	}	
	CdamFreeDevice(buffer, num_tet * sizeof(f64));
	CdamFreeDevice(elem_invJ, num_tet * sizeof(f64) * 10);
	CdamFreeDevice(shgradl, num_tet * sizeof(f64) * NSHL * 3);

	CdamFreeDevice(elem_F, num_tet * sizeof(f64) * NSHL);
	CdamFreeDevice(elem_J, num_tet * sizeof(f64) * NSHL * NSHL);

	CdamFreeDevice(qr_wgalpha, 2 * num_tet * sizeof(f64) * 4);
	CdamFreeDevice(qr_wggradalpha, 2 * num_tet * sizeof(f64) * 3);
	cublasDestroy(handle);
}

__END_DECLS__
