// #include <cuda_profiler_api.h>

#include "alloc.h"
#include "json.h"
#include "Mesh.h"
#include "csr.h"
#include "parallel_matrix.h"

#include "assemble.h"

#include "form.h"

#define kRHOC (0.5)
#define kDT (5e-2)
#define kALPHAM (0.5 * (3.0 - kRHOC) / (1.0 + kRHOC))
#define kALPHAF (1.0 / (1.0 + kRHOC))
#define kGAMMA (0.5 + kALPHAM - kALPHAF)

#define NQR (4)
// #define BS (6)
#define M2D(aa, ii) ((aa) * BS + (ii))
#define M4D(aa, bb, ii, jj) \
	(((aa) * (NSHL) + bb) * (BS * BS) + (ii) * BS + (jj))

#define kRHO (1.0e3)
/// #define kCP (4.2e3)
#define kCP (1.0)
#define kKAPPA (0.66)
/// #define kKAPPA (0.0)
#define kMU (10.0/3.0)


__BEGIN_DECLS__
void LoadElementValue(index_type batch_size, index_type* batch_index_ptr,
											index_type* ien, index_type nshl, index_type elem_size,
											void* x, index_type xstride,
											void* y, index_type ystride); 
void GetElemInvJ3D(index_type batch_size, index_type* batch_index_ptr,
									 index_type* ien, value_type* coord, value_type* elem_metric, Arena scratch);
void GetShapeGrad(index_type num_elem, value_type* elem_invJ, value_type* shgrad);

void IntElemAssembly(index_type num_elem, value_type* elem_invJ, value_type* shgrad,
											value_type* qr_wgalpha, value_type* qr_dwgalpha, value_type* qr_wggradalpha,
											value_type* elem_F, value_type* elem_J, FEMOptions opt, index_type flag);
void ElemRHSLocal2Global(index_type num_elem, index_type* batch_index_ptr, index_type* ien,
												 index_type elem_size, value_type* elem_F, index_type Fstride,
												 value_type* F, index_type Fstride_global, const index_type* map);

void AddElemValueToVec(value_type* b, index_type nelem, index_type nshl, index_type* ien,
											 CdamLayout* map, value_type* elem_F) {
#if CDAM_USE_CUDA
#else
#endif
}

// void MatrixAddElementLHS(CdamParMat* J, index_type nshl, index_type bs, index_type num_elem, index_type* ien,
// 												 index_type* batch_index_ptr, value_type* elem_J, index_type Jstride);


static void ParseFEMOptions(FEMOptions opt, cJSON* config) {
	value_type rhoc;
	CdamMemset(opt, 0, MAX_NUM_OPTIONS * sizeof(value_type), HOST_MEM);

	rhoc = (value_type)JSONGetItem(config, "VMS.RHOC")->valuedouble;
	opt[OPTION_RHOC] = rhoc;
	opt[OPTION_ALPHA_M] = 0.5 * (3.0 - rhoc) / (1.0 + rhoc);
	opt[OPTION_ALPHA_F] = 1.0 / (1.0 + rhoc);
	opt[OPTION_GAMMA] = 0.5 + opt[OPTION_ALPHA_M] - opt[OPTION_ALPHA_F];

	opt[OPTION_DT] = (value_type)JSONGetItem(config, "Physics.dt")->valuedouble;
	if(cJSON_IsTrue(JSONGetItem(config, "VMS.IncompressibleNS.Included"))) {
		opt[OPTION_NS] = 1.0;
		opt[OPTION_NS_SUPG] = cJSON_IsTrue(JSONGetItem(config, "VMS.IncompressibleNS.SUPG")) ? 1.0 : 0.0;
		opt[OPTION_NS_PSPG] = cJSON_IsTrue(JSONGetItem(config, "VMS.IncompressibleNS.PSPG")) ? 1.0 : 0.0;
		opt[OPTION_NS_LSIC] = cJSON_IsTrue(JSONGetItem(config, "VMS.IncompressibleNS.LSIC")) ? 1.0 : 0.0;
		opt[OPTION_NS_OTHER] = cJSON_IsTrue(JSONGetItem(config, "VMS.IncompressibleNS.OtherVMS")) ? 1.0 : 0.0;
		opt[OPTION_NS_TAUBAR] = cJSON_IsTrue(JSONGetItem(config, "VMS.IncompressibleNS.TauBar")) ? 1.0 : 0.0;
		opt[OPTION_NS_KDC] = cJSON_IsTrue(JSONGetItem(config, "VMS.IncompressibleNS.KDC"));
	}

	if(cJSON_IsTrue(JSONGetItem(config, "VMS.Temperature.Included"))) {
		opt[OPTION_T] = 1.0;
		opt[OPTION_T_SUPG] = cJSON_IsTrue(JSONGetItem(config, "VMS.Temperature.SUPG")) ? 1.0 : 0.0;
		opt[OPTION_T_KDC] = JSONGetItem(config, "VMS.Temperature.KDC")->valuedouble;
	}

	if(cJSON_IsTrue(JSONGetItem(config, "VMS.Levelset.Included"))) {
		opt[OPTION_PHI] = 1.0;
		opt[OPTION_PHI_SUPG] = cJSON_IsTrue(JSONGetItem(config, "VMS.Levelset.SUPG")) ? 1.0 : 0.0;
		opt[OPTION_PHI_KDC] = JSONGetItem(config, "VMS.Levelset.KDC")->valuedouble;
	}

}
static 
const f64 h_shlu[16] = {0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105,
											0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105,
											0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105,
											0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685};

void AssembleSystemTetra(index_type num_node, value_type* coord,
												 index_type num_tet, index_type* ien,
												 index_type num_color, index_type* color_batch_offset, index_type* color_batch_ind,
												 value_type* wgalpha_dptr, value_type* dwgalpha_dptr,
												 value_type* F, Matrix* J,
												 FEMOptions opt, Arena scratch) {
	const i32 NSHL = 4;
	// value_type* d_shlu = CdamTMalloc(value_type, NQR * NSHL, DEVICE_MEM);
	value_type one = 1.0, zero = 0.0, minus_one = -1.0;

	value_type* buffer, *elem_invJ, *shgradg, *qr_wgalpha, *qr_dwgalpha, *qr_wggradalpha;
	int* pivot, *info;
	value_type* elem_F = NULL;
	value_type* elem_J = NULL;

	value_type* d_shlu = (value_type*)ArenaPush(sizeof(value_type), NQR * NSHL, &scratch, 0);
	CdamMemcpy(d_shlu, h_shlu, NQR * NSHL * sizeof(value_type), DEVICE_MEM, HOST_MEM);


	index_type c, color_batch_size;
	index_type* color_batch_index_ptr;

	index_type bs = 0;
	bs += 3 * opt[OPTION_NS] + opt[OPTION_T] + opt[OPTION_PHI];


	Arena scratch_original = scratch;

	for(c = 0; c < num_color; c++) {
		/* Get the batch size and content */
		color_batch_size = color_batch_offset[c + 1] - color_batch_offset[c];

		color_batch_index_ptr = color_batch_ind + color_batch_offset[c];

		index_type* ien_color = (index_type*)ArenaPush(sizeof(index_type), color_batch_size * NSHL, &scratch, 0);

		RestrictVec(ien_color, ien, color_batch_size, color_batch_index_ptr, NSHL * sizeof(index_type));


		/* Calculate the element metrics */
		elem_invJ = (value_type*)ArenaPush(sizeof(value_type), color_batch_size * (3 * 3 + 1), &scratch, 0);

		GetElemInvJ3D(color_batch_size, color_batch_index_ptr,
									ien, coord, elem_invJ, scratch);

		shgradg = (value_type*)ArenaPush(sizeof(value_type), color_batch_size * NSHL * 3, &scratch, 0);
		GetShapeGrad(color_batch_size, elem_invJ, shgradg);

		dgemmStridedBatched(BLAS_T, BLAS_N,
							3, 3, 3,
							one,
							shgradg + 3, 3, NSHL * 3,
							shgradg + 3, 3, NSHL * 3,
							zero,
							elem_invJ, 3, NSHL * 3,
							color_batch_size);

		buffer = (value_type*)ArenaPush(sizeof(value_type), color_batch_size * NSHL * bs, &scratch, 0);
		/* Interpolate the field values */
		RestrictAddVecStrided(buffer + 0 * NSHL, bs * NSHL, wgalpha_dptr + 0 * num_node, 3,
													color_batch_size * NSHL, ien_color, 1);


		LoadElementValue(color_batch_size, color_batch_index_ptr, ien,
										 NSHL, 3 * sizeof(value_type),
										 wgalpha_dptr, 3, buffer, bs * NSHL);
		LoadElementValue(color_batch_size, color_batch_index_ptr, ien,
										 NSHL, sizeof(value_type),
										 dwgalpha_dptr + num_node * 3, 1, buffer + 3 * NSHL, bs * NSHL);
		LoadElementValue(color_batch_size, color_batch_index_ptr, ien,
										 NSHL, sizeof(value_type),
										 dwgalpha_dptr + num_node * 4, 1, buffer + 4 * NSHL, bs * NSHL);
		LoadElementValue(color_batch_size, color_batch_index_ptr, ien,
										 NSHL, sizeof(value_type),
										 dwgalpha_dptr + num_node * 5, 1, buffer + 5 * NSHL, bs * NSHL);

		qr_wggradalpha = (value_type*)ArenaPush(sizeof(value_type), color_batch_size * 3 * bs, &scratch, 0);
		dgemmStridedBatched(BLAS_N, BLAS_N,
												3, bs, NSHL,
												one,
												shgradg, 3, NSHL* 3,
												buffer, NSHL, NSHL * bs,
												zero,
												qr_wggradalpha, 3, bs * 3,
												color_batch_size);
		 
		qr_wgalpha = (value_type*)ArenaPush(sizeof(value_type), color_batch_size * NQR * bs, &scratch, 0);
		dgemm(BLAS_N, BLAS_N,
					NQR, color_batch_size * bs, NSHL,
					one,
					d_shlu, NQR,
					buffer, NSHL,
					zero,
					qr_wgalpha, NQR);

		LoadElementValue(color_batch_size, color_batch_index_ptr, ien,
										 NSHL, 3 * sizeof(value_type),
										 dwgalpha_dptr, 3, buffer, bs * NSHL);
		LoadElementValue(color_batch_size, color_batch_index_ptr, ien,
										 NSHL, sizeof(value_type),
										 dwgalpha_dptr + num_node * 3, 1, buffer + 3 * NSHL, bs * NSHL);
		LoadElementValue(color_batch_size, color_batch_index_ptr, ien,
										 NSHL, sizeof(value_type),
										 dwgalpha_dptr + num_node * 4, 1, buffer + 4 * NSHL, bs * NSHL);
		LoadElementValue(color_batch_size, color_batch_index_ptr, ien,
										 NSHL, sizeof(value_type),
										 dwgalpha_dptr + num_node * 5, 1, buffer + 5 * NSHL, bs * NSHL);

		qr_dwgalpha = (value_type*)ArenaPush(sizeof(value_type), color_batch_size * NQR * bs, &scratch, 0);
		dgemm(BLAS_T, BLAS_N,
					NQR, color_batch_size * bs, NSHL,
					one,
					d_shlu, NQR,
					buffer, NSHL,
					zero,
					qr_dwgalpha, NQR);
		ArenaPop(sizeof(value_type), color_batch_size * NSHL * bs, &scratch);

		if(F) {
			elem_F = (value_type*)ArenaPush(sizeof(value_type), color_batch_size * NSHL * bs, &scratch, 0);
			IntElemAssembly(color_batch_size, elem_invJ, shgradg,
											qr_wgalpha, qr_dwgalpha, qr_wggradalpha,
											elem_F, NULL, opt, 0);
			AddElemValueToVec(F, color_batch_size, NSHL, ien_color, NULL, elem_F);
			/*
			ElemRHSLocal2Global(color_batch_size, color_batch_index_ptr, ien,
													3, elem_F, bs,
													F, 3, (const index_type*)NULL);
			ElemRHSLocal2Global(color_batch_size, color_batch_index_ptr, ien,
													1, elem_F + 3, bs,
													F + num_node * 3, 1, (const index_type*)NULL);
			ElemRHSLocal2Global(color_batch_size, color_batch_index_ptr, ien,
													1, elem_F + 4, bs,
													F + num_node * 4, 1, (const index_type*)NULL);
			ElemRHSLocal2Global(color_batch_size, color_batch_index_ptr, ien,
													1, elem_F + 5, bs,
													F + num_node * 5, 1, (const index_type*)NULL);
													*/
		}
		if(J) {
			elem_J = (value_type*)ArenaPush(sizeof(value_type), color_batch_size * NSHL * NSHL * bs * bs, &scratch, 0);
			IntElemAssembly(color_batch_size, elem_invJ, shgradg,
											qr_wgalpha, qr_dwgalpha, qr_wggradalpha,
											NULL, elem_J, opt, 0);
			// MatrixAddElementLHS(J, NSHL, bs, color_batch_size, ien, color_batch_index_ptr, elem_J, bs * NSHL);
			CdamParMatAddElemValueBatched(J, color_batch_size, NSHL, ien_color, elem_J, scratch);
		}

		scratch = scratch_original;
	}

}

void AssembleSystem(void* mesh, void* wgalpha, void* dwgalpha,
										void* F, void* J, cJSON* config, Arena scratch) {
	index_type num_node = CdamMeshNumNode(mesh);
	value_type* coord = CdamMeshCoord(mesh);
	index_type num_tet = CdamMeshNumTet(mesh);
	index_type num_prism = CdamMeshNumPrism(mesh);
	index_type num_hex = CdamMeshNumHex(mesh);
	index_type* ien = CdamMeshIEN(mesh);

	index_type ec, num_color = CdamMeshNumColor(mesh);
	index_type* color_batch_offset = CdamMeshColorBatchOffset(mesh);
	index_type* color_batch_ind = CdamMeshColorBatchInd(mesh);
	index_type bs = 0;
	FEMOptions opt;
	ParseFEMOptions(opt, config);

	bs += 3 * opt[OPTION_NS] + opt[OPTION_T] + opt[OPTION_PHI];

	if (F) {
		CdamMemset(F, 0, num_node * bs * sizeof(value_type), DEVICE_MEM);
	}
	if (J) {
		// MatrixZero(J);
		CdamParMatZero(J);
	}

	if (num_tet) {
		AssembleSystemTetra(num_node, coord, num_tet, ien,
												num_color, color_batch_offset, color_batch_ind,
												wgalpha, dwgalpha, F, J, opt, scratch);
	}

	if (num_prism) {
		/*
		AssembleSystemPrism(num_node, coord, num_prism, ien + num_tet * 4,
												wgalpha, dwgalpha, F, J, opt, arena);
		AssembleSystemPrismFace(...);
		*/
	}

	if (num_hex) {
		/*
		AssembleSystemHex(num_node, coord, num_hex, ien + num_tet * 4 + num_prism * 6,
											wgalpha, dwgalpha, F, J, opt, arena);
		AssembleSystemHexFace(...);
		*/
	}
}

__END_DECLS__
