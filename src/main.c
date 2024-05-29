#include <stdio.h>
#include <time.h>
#include <math.h>
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>

#include "alloc.h"
#include "h5util.h"
#include "Mesh.h"
#include "Field.h"
#include "Particle.h"
#include "csr.h"
#include "krylov.h"
#include "assemble.h"

#define kRHOC (0.5)
#define kDT (1e-1)
#define kALPHAM ((3.0 - kRHOC) / (1.0 + kRHOC))
#define kALPHAF (1.0 / (1.0 + kRHOC))
#define kGAMMA (0.5 + kALPHAM - kALPHAF)

#define BS (6)

void AssembleSystem(Mesh3D* mesh, f64* wgalpha, f64* dwgalpha, f64* F, Matrix* J) {
	// u32 i, j, k;
	// u32 num_node = (u32)Mesh3DDataNumNode(Mesh3DHost(mesh));
	u32 num_tet = (u32)Mesh3DDataNumTet(Mesh3DHost(mesh));
	u32 num_prism = (u32)Mesh3DDataNumPrism(Mesh3DHost(mesh));
	u32 num_hex = (u32)Mesh3DDataNumHex(Mesh3DHost(mesh));
	// f64* xg = Mesh3DDataCoord(Mesh3DHost(mesh));

	if(num_tet) {
		AssembleSystemTet(mesh, wgalpha, dwgalpha, F, J);
	}
	
	if(num_prism) {
	}

	if(num_hex) {
	}
}

void SolveFlowSystem(Mesh3D* mesh, f64* wgold, f64* dwgold, f64* dwg, Matrix* J, f64* F, f64* dx, Krylov* ksp) {
	u32 maxit = 4, iter = 0;
	f64 tol = 1.0e-5;
	b32 converged = FALSE;
	f64 rnorm, rnorm_init;
	f64 minus_one = -1.0;
	clock_t start, end;

	f64 fact1[] = {1.0 - kALPHAM, kALPHAM, kALPHAM - 1.0, -kALPHAM};
	f64 fact2[] = {kDT * kALPHAF * (1.0 - kGAMMA), kDT * kALPHAF * kGAMMA,
								 -kDT * kALPHAF * (1.0 - kGAMMA), -kDT * kALPHAF * kGAMMA};

	cublasHandle_t handle;
	cublasCreate(&handle);

	u32 num_node = Mesh3DDataNumNode(Mesh3DHost(mesh));

	f64* wgalpha = (f64*)CdamMallocDevice(num_node * sizeof(f64) * BS);
	f64* dwgalpha = (f64*)CdamMallocDevice(num_node * sizeof(f64) * BS);

	/* dwgalpha = fact1[0] * dwgold + fact[1] * dwg */ 
	cudaMemset(dwgalpha, 0, num_node * sizeof(f64) * BS);
	cublasDaxpy(handle, num_node * BS, fact1 + 0, dwgold, 1, dwgalpha, 1);
	cublasDaxpy(handle, num_node * BS, fact1 + 1, dwg, 1, dwgalpha, 1);
	cublasDcopy(handle, num_node, dwg + num_node * 3, 1, dwgalpha + num_node * 3, 1);

	/* wgalpha = wgold + fact2[0] * dwgold + fact2[1] * dwg */
	cublasDcopy(handle, num_node * BS, wgold, 1, wgalpha, 1);
	cublasDaxpy(handle, num_node * BS, fact2 + 0, dwgold, 1, wgalpha, 1);
	cublasDaxpy(handle, num_node * BS, fact2 + 1, dwg, 1, wgalpha, 1);
	cudaMemset(wgalpha + num_node * 3, 0, num_node * sizeof(f64));

	/* Construct the right-hand side */
	
	start = clock();
	AssembleSystem(mesh, wgalpha, dwgalpha, F, NULL);
	end = clock();
	fprintf(stdout, "Assemble time: %f ms\n", (f64)(end - start) / CLOCKS_PER_SEC * 1000.0);
	cublasDnrm2(handle, num_node * BS, F, 1, &rnorm_init);


	CUGUARD(cudaGetLastError());

	fprintf(stdout, "Newton %d) abs = %6.4e rel = %6.4e (tol = %6.4e)\n",
					0, rnorm_init, 1.0, tol);

	while(!converged && iter < maxit) {
		/* Construct the Jacobian matrix */
		start = clock();
		AssembleSystem(mesh, wgalpha, dwgalpha, NULL, J);
		end = clock();
		fprintf(stdout, "Assemble J time: %f ms\n", (f64)(end - start) / CLOCKS_PER_SEC * 1000.0);
		CUGUARD(cudaGetLastError());
		
	
		/* Solve the linear system */
		cudaMemset(dx, 0, num_node * sizeof(f64));
		start = clock();
		KrylovSolve(ksp, J, dx, F);
		end = clock();
		fprintf(stdout, "Krylov solve time: %f ms\n", (f64)(end - start) / CLOCKS_PER_SEC * 1000.0);

		/* Update the solution */	
		// cublasDaxpy(handle, ArrayLen(d_dwg), &minus_one, dx, 1, ArrayData(d_dwg), 1);
		/* Update dwg by dwg -= dx */
		cublasDaxpy(handle, num_node * BS, &minus_one, dx, 1, dwg, 1);
		/* Update dwgalpha by dwgalpha[[0,1,2], :] -= kALPHAM * dx[[0,1,2], :] */ 
		cublasDaxpy(handle, num_node * 3, fact1 + 3, dx, 1, dwgalpha, 1);
		/* Update dwgalpha by dwgalpha[3, :] -= dx[3, :] */ 
		cublasDaxpy(handle, num_node, &minus_one, dx + num_node * 3, 1, dwgalpha + num_node * 3, 1);
		/* Update dwgalpha by dwgalpha[[4,5], :] -= kALPHAM * dx[[4,5], :] */ 
		cublasDaxpy(handle, num_node * 2, fact1 + 3, dx + num_node * 4, 1, dwgalpha + num_node * 4, 1);
		/* Update wgalpha by wgalpha[[0,1,2], :] -= kDT * kALPHAF * kGAMMA * dx[[0,1,2], :] */
		cublasDaxpy(handle, num_node * 3, fact2 + 3, dx, 1, wgalpha, 1);
		/* Update wgalpha by wgalpha[[4,5], :] -= kDT * kALPHAF * kGAMMA * dx[[4,5], :] */
		cublasDaxpy(handle, num_node * 2, fact2 + 3, dx + num_node * 4, 1, wgalpha + num_node * 4, 1);


		start = clock();
		AssembleSystem(mesh, wgalpha, dwgalpha, F, NULL);
		end = clock();
		fprintf(stdout, "Assemble F time: %f ms\n", (f64)(end - start) / CLOCKS_PER_SEC * 1000.0);
		cublasDnrm2(handle, num_node * BS, F, 1, &rnorm);
		fprintf(stdout, "Newton %d) abs = %6.4e rel = %6.4e (tol = %6.4e)\n",
						iter + 1, rnorm, rnorm / (rnorm_init + 1e-16), tol);
		if (rnorm < tol * (rnorm_init + 1e-16)) {
			converged = TRUE;
		}
		iter++;

	}

	CdamFreeDevice(dwgalpha, num_node * sizeof(f64) * BS);
	CdamFreeDevice(wgalpha, num_node * sizeof(f64) * BS);
	cublasDestroy(handle);
}


void MyFieldInit(f64* value, void* ctx) {
	double eps = 1.5e-4 * 0.5;
	double z, h;
	Mesh3D* mesh = (Mesh3D*)ctx;
	Mesh3DData* data = Mesh3DHost(mesh);
	u32 i;
	u32 num_node = Mesh3DDataNumNode(data);
	f64* coord = Mesh3DDataCoord(data);

	for(i = 0; i < num_node; i++) {
		value[i * 3 + 0] = coord[i * 3 + 0];
		value[i * 3 + 1] = coord[i * 3 + 1];
		value[i * 3 + 2] = coord[i * 3 + 2];
	}
	for(i = 0; i < num_node; ++i) {
		z = 2e-4 - coord[i * 3 + 2];
		if (z > eps) {
			h = 1.0;
		}
		else if (z < -eps) {
			h = 0.0;
		}
		else {
			h = 0.5 * (1.0 + z / eps + sin(M_PI * z / eps) / M_PI);
		}
		value[num_node * 3 + i] = 0.0;
		value[num_node * 4 + i] = h;
		value[num_node * 5 + i] = h;
	}
}


int main() {
	b32 converged = FALSE;
	i32 step = 0, num_step = 1;
	char filename_buffer[256] = {0};
	struct cudaDeviceProp prop;
	i32 num_device;
	cudaGetDeviceCount(&num_device);
	ASSERT(num_device > 0 && "No CUDA device found");
	if (num_device) {
		cudaGetDeviceProperties(&prop, 0);
		printf("==================================================\n");
		printf(" * Device name: %s\n", prop.name);
		printf(" * Compute capability: %d.%d\n", prop.major, prop.minor);
		printf(" * Total global memory: %f (MB)\n", (f64)prop.totalGlobalMem / 1024.0 / 1024.0);
		printf(" * Shared memory per block: %f (KB)\n", (f64)prop.sharedMemPerBlock / 1024.0);
		printf(" * Registers per block: %f (KB)\n", (f64)prop.regsPerBlock / 1024.0);
		printf(" * Warp size: %d\n", prop.warpSize);
		printf(" * Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf(" * Max threads dimension: %d X %d X %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf(" * Max grid size: %d X %d X %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf(" * Clock rate: %f (Hz) \n", (f64)prop.clockRate);
		printf(" * Total constant memory: %f (MB)\n", (f64)prop.totalConstMem / 1024.0 / 1024.0);
		printf(" * Peak memory clock rate: %f (MHz)\n", (f64)prop.memoryClockRate / 1e6);
		printf(" * Memory bandwidth: %f (GB/s)\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
		printf("==================================================\n");
	}

	H5FileInfo* h5_handler = H5OpenFile("cube.h5", "r");
	Mesh3D* mesh = Mesh3DCreateH5(h5_handler, "mesh");
	H5CloseFile(h5_handler);

	cublasHandle_t handle;
	cublasCreate(&handle);
	u32 num_node = Mesh3DNumNode(mesh);

	index_type n_offset = 5;
	index_type offset[] = {0, 3, 4, 5, 6};

	CSRAttr* spy1x1 = CSRAttrCreate(mesh);
	CSRAttr* spy1x3 = CSRAttrCreateBlock(spy1x1, 1, 3);
	CSRAttr* spy3x1 = CSRAttrCreateBlock(spy1x1, 3, 1);
	CSRAttr* spy3x3 = CSRAttrCreateBlock(spy1x1, 3, 3);
	Matrix* J = MatrixCreateTypeNested(n_offset, offset);
	MatrixNested* mat_nested = (MatrixNested*)J->data;
	mat_nested->csr_auxiliary = spy1x1;

	mat_nested->mat[4*0+0] = MatrixCreateTypeCSR(spy3x3);
	mat_nested->mat[4*0+1] = MatrixCreateTypeCSR(spy3x1);
	mat_nested->mat[4*0+2] = MatrixCreateTypeCSR(spy3x1);
	mat_nested->mat[4*0+3] = MatrixCreateTypeCSR(spy3x1);

	mat_nested->mat[4*1+0] = MatrixCreateTypeCSR(spy1x3);
	mat_nested->mat[4*1+1] = MatrixCreateTypeCSR(spy1x1);
	mat_nested->mat[4*1+2] = MatrixCreateTypeCSR(spy1x1);
	mat_nested->mat[4*1+3] = MatrixCreateTypeCSR(spy1x1);

	mat_nested->mat[4*2+0] = MatrixCreateTypeCSR(spy1x3);
	mat_nested->mat[4*2+1] = MatrixCreateTypeCSR(spy1x1);
	mat_nested->mat[4*2+2] = MatrixCreateTypeCSR(spy1x1);
	mat_nested->mat[4*2+3] = MatrixCreateTypeCSR(spy1x1);

	mat_nested->mat[4*3+0] = MatrixCreateTypeCSR(spy1x3);
	mat_nested->mat[4*3+1] = MatrixCreateTypeCSR(spy1x1);
	mat_nested->mat[4*3+2] = MatrixCreateTypeCSR(spy1x1);
	mat_nested->mat[4*3+3] = MatrixCreateTypeCSR(spy1x1);

	Krylov* ksp = KrylovCreateGMRES(120, 1e-12, 1e-5);


	time_t start, end;
	start = time(NULL);
	Mesh3DGenerateColorBatch(mesh);
	end = time(NULL);
	fprintf(stdout, "Coloring time: %f s\n", difftime(end, start));
	if(0){
		// Mesh3DData* dev = Mesh3DHost(mesh);
		// u32 num_elem = dev->num_tet;
		// color_t* h_color = (color_t*)CdamMallocHost(num_elem * sizeof(color_t));
		// color_t* d_color = mesh->color;
		// cudaMemcpy(h_color, d_color, num_elem * sizeof(color_t), D2H);
		// FILE* fp = fopen("color.txt", "w");
		// for (u32 i = 0; i < num_elem; i++) {
		// 	fprintf(fp, "%d\n", h_color[i]);
		// }
		// return 0;
	}


	// Field* wgold = FieldCreate3D(mesh, 1);
	// Field* dwgold = FieldCreate3D(mesh, 1);
	// Field* dwg = FieldCreate3D(mesh, 1);
	// FieldUpdateDevice(wgold);
	// FieldUpdateDevice(dwgold);
	// FieldUpdateDevice(dwg);
	f64* wgold = (f64*)CdamMallocDevice(num_node * sizeof(f64) * BS);
	f64* dwgold = (f64*)CdamMallocDevice(num_node * sizeof(f64) * BS);
	f64* dwg = (f64*)CdamMallocDevice(num_node * sizeof(f64) * BS);
	f64* buffer = (f64*)CdamMallocHost(num_node * sizeof(f64) * BS);

	f64* F = (f64*)CdamMallocDevice(num_node * sizeof(f64) * BS);
	f64* dx = (f64*)CdamMallocDevice(num_node * sizeof(f64) * BS);

	cudaMemset(wgold, 0, num_node * sizeof(f64) * BS);
	cudaMemset(dwgold, 0, num_node * sizeof(f64) * BS);
	cudaMemset(dwg, 0, num_node * sizeof(f64) * BS);
	memset(buffer, 0, num_node * sizeof(f64) * BS);

	ParticleContext* pctx = ParticleContextCreate(100);
	ParticleContextUpdateDevice(pctx);
	if (step) {
		sprintf(filename_buffer, "sol.%d.h5", step);
		h5_handler = H5OpenFile(filename_buffer, "r");
		// FieldLoad(wgold, h5_handler, "w");
		// FieldLoad(dwgold, h5_handler, "dw");
		memset(buffer, 0, num_node * sizeof(f64) * BS);
		H5ReadDatasetf64(h5_handler, "u", buffer);
		H5ReadDatasetf64(h5_handler, "phi", buffer + num_node * 4);
		H5ReadDatasetf64(h5_handler, "T", buffer + num_node * 5);
		cudaMemcpy(wgold, buffer, num_node * sizeof(f64) * 3, cudaMemcpyHostToDevice);

		memset(buffer, 0, num_node * sizeof(f64) * BS);
		H5ReadDatasetf64(h5_handler, "du", buffer);
		H5ReadDatasetf64(h5_handler, "p", buffer + num_node * 3);
		H5ReadDatasetf64(h5_handler, "phi", buffer + num_node * 4);
		H5ReadDatasetf64(h5_handler, "T", buffer + num_node * 5);
		cudaMemcpy(dwgold, buffer, num_node * sizeof(f64) * BS, cudaMemcpyHostToDevice);

		// ParticleContextLoad(pctx, h5_handler, "ptc/test/group/context");
		
		cudaMemcpy(dwg, dwgold, num_node * sizeof(f64) * BS, cudaMemcpyDeviceToDevice);

		H5CloseFile(h5_handler);
	}
	else {
		/* Initial condition */
		// FieldInit(wgold, MyFieldInit, mesh);
		MyFieldInit(buffer, mesh);
		/* wgold[0:3*N] = buffer[0:3*N] */
		/* wgold[4*N:6*N] = buffer[4*N:6*N] */
		/* dwgold[0:3*N] = 0 */
		/* dwgold[4*N:5*N] = buffer[4*N:5*N] */
		/* dwgold[5*N:6*N] = 0 */
		cudaMemcpy(wgold, buffer, num_node * sizeof(f64) * BS, cudaMemcpyHostToDevice);
		cudaMemset(wgold + num_node * 3, 0, num_node * sizeof(f64));
		cudaMemcpy(dwgold + num_node * 3, buffer + num_node * 3, num_node * sizeof(f64), cudaMemcpyHostToDevice);
	
		h5_handler = H5OpenFile("sol.0.h5", "w");
		H5WriteDatasetf64(h5_handler, "u", num_node * 3, buffer);
		H5WriteDatasetf64(h5_handler, "phi", num_node, buffer + num_node * 4);
		H5WriteDatasetf64(h5_handler, "T", num_node, buffer + num_node * 5);

		memset(buffer, 0, num_node * sizeof(f64) * BS);
		H5WriteDatasetf64(h5_handler, "du", num_node * 3, buffer);
		H5WriteDatasetf64(h5_handler, "p", num_node * 3, buffer + num_node * 3);
		H5WriteDatasetf64(h5_handler, "dphi", num_node, buffer + num_node * 4);
		H5WriteDatasetf64(h5_handler, "dT", num_node, buffer + num_node * 5);

		H5CloseFile(h5_handler);
	}
	return 0;

	f64 fac_pred[] = {(kGAMMA - 1.0) / kGAMMA};
	f64 fac_corr[] = {kDT * (1.0 - kGAMMA), kDT * kGAMMA};
	while(step++ < num_step) {
		/* Prediction stage */
		// ArrayScale(FieldDevice(dwg), (kGAMMA - 1.0) / kGAMMA);
		cublasDscal(handle, num_node * 3, fac_pred, dwg, 1);
		cublasDscal(handle, num_node * 2, fac_pred, dwg + num_node * 4, 1);

		/* Generate new particles */
		// ParticleContextAdd(pctx);

		/* Newton-Raphson iteration */
		converged = FALSE;
		while(!converged) {
			SolveFlowSystem(mesh, wgold, dwgold, dwg, J, F, dx, ksp);
#ifdef DEBUG
			SolveParticleSystem(pctx);
#endif
			converged = TRUE;
		}

		/* Update stage */
		cublasDaxpy(handle, num_node * 3, fac_corr + 0, dwgold, 1, wgold, 1);
		cublasDaxpy(handle, num_node * 2, fac_corr + 0, dwgold + num_node * 4, 1, wgold + num_node * 4, 1);
		cublasDaxpy(handle, num_node * 3, fac_corr + 1, dwg, 1, wgold, 1);
		cublasDaxpy(handle, num_node * 2, fac_corr + 1, dwg + num_node * 4, 1, wgold + num_node * 4, 1);
		cublasDcopy(handle, num_node * 6, dwg, 1, dwgold, 1);
		/* Particle update */
		// ParticleContextUpdate(pctx);
		/* Particle removal */
		// ParticleContextRemove(pctx);

		if (step % 1 == 0) {
			sprintf(filename_buffer, "sol.%d.h5", step);
			h5_handler = H5OpenFile(filename_buffer, "w");
		
			cudaMemcpy(buffer, wgold, num_node * sizeof(f64) * BS, cudaMemcpyDeviceToHost);
			H5WriteDatasetf64(h5_handler, "u", num_node * 3, buffer);
			H5WriteDatasetf64(h5_handler, "phi", num_node, buffer + num_node * 4);
			H5WriteDatasetf64(h5_handler, "T", num_node, buffer + num_node * 5);

			cudaMemcpy(buffer, dwgold, num_node * sizeof(f64) * BS, cudaMemcpyDeviceToHost);
			H5WriteDatasetf64(h5_handler, "du", num_node * 3, buffer);
			H5WriteDatasetf64(h5_handler, "p", num_node, buffer + num_node * 3);
			H5WriteDatasetf64(h5_handler, "phi", num_node, buffer + num_node * 4);
			H5WriteDatasetf64(h5_handler, "T", num_node, buffer + num_node * 5);
			
			// ParticleContextUpdateHost(pctx);
			// ParticleContextSave(pctx, h5_handler, "ptc/test/group/context");
			H5CloseFile(h5_handler);
		}	
	}


	ParticleContextDestroy(pctx);
	CdamFreeDevice(wgold, num_node * sizeof(f64) * BS);
	CdamFreeDevice(dwgold, num_node * sizeof(f64) * BS);
	CdamFreeDevice(dwg, num_node * sizeof(f64) * BS);
	cublasDestroy(handle);
	MatrixDestroy(J);
	// FieldDestroy(wgold);
	// FieldDestroy(dwgold);
	// FieldDestroy(dwg);
	Mesh3DDestroy(mesh);
	return 0;
}
