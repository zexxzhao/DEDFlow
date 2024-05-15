#include <stdio.h>
#include <math.h>
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "alloc.h"
#include "h5util.h"
#include "Mesh.h"
#include "Field.h"
#include "Particle.h"
#include "csr.h"
#include "krylov.h"
#include "assemble.h"

#define kRHOC (0.5)
#define kDT (0.1)
#define kALPHAM ((3.0 - kRHOC) / (1.0 + kRHOC))
#define kALPHAF (1.0 / (1.0 + kRHOC))
#define kGAMMA (0.5 + kALPHAM - kALPHAF)


void AssembleSystem(Mesh3D* mesh, Field* wgold, Field* dwgold, Field* dwg, f64* F, Matrix* J) {
	u32 i, j, k;
	u32 num_node = (u32)Mesh3DDataNumNode(Mesh3DHost(mesh));
	u32 num_tet = (u32)Mesh3DDataNumTet(Mesh3DHost(mesh));
	u32 num_prism = (u32)Mesh3DDataNumPrism(Mesh3DHost(mesh));
	u32 num_hex = (u32)Mesh3DDataNumHex(Mesh3DHost(mesh));
	f64* xg = Mesh3DDataCoord(Mesh3DHost(mesh));

	if(num_tet) {
		AssembleSystemTet(mesh, wgold, dwgold, dwg, F, J);
	}
	
	if(num_prism) {
		// AssembleSystem_prism(mesh, wgold, dwgold, dwg, F, J);
	}

	if(num_hex) {
		// AssembleSystem_hex(mesh, wgold, dwgold, dwg, F, J);
	}
}

void SolveFlowSystem(Mesh3D* mesh, Field* wgold, Field* dwgold, Field* dwg) {
	u32 maxit = 10;
	f64 tol = 1.0e-6;
	b32 converged = FALSE;
	f64 rnorm, rnorm_init;
	const f64 one = 1.0, zero = 0.0, minus_one = -1.0;

	u32 num_node = Mesh3DDataNumNode(Mesh3DHost(mesh));
	f64* F = (f64*)CdamMallocDevice(num_node * sizeof(f64));
	f64* dx = (f64*)CdamMallocDevice(num_node * sizeof(f64));
	CSRAttr* spy = CSRAttrCreate(mesh);
	Matrix* J = MatrixCreateTypeCSR(spy);
	Krylov* ksp = KrylovCreateGMRES(60, tol, tol);

	Array* d_dwg = FieldDevice(dwg);
	cublasHandle_t handle;

	/* Construct the right-hand side */
	AssembleSystem(mesh, wgold, dwgold, dwg, F, NULL);
	cublasCreate(&handle);
	cublasDnrm2(handle, num_node, F, 1, &rnorm_init);

	while(!converged && maxit--) {
		/* Construct the Jacobian matrix */
		AssembleSystem(mesh, wgold, dwgold, dwg, NULL, J);
	
		/* Solve the linear system */
		KrylovSolve(ksp, J, F, dx);	

		/* Update the solution */	
		// ArrayAXPY(FieldDevice(dwg), -1.0, dx);
		cublasDaxpy(handle, ArrayLen(d_dwg), &minus_one, dx, 1, ArrayData(d_dwg), 1);

		/* Construct the right-hand side */
		AssembleSystem(mesh, wgold, dwgold, dwg, F, NULL);
		cublasDnrm2(handle, num_node, F, 1, &rnorm);
		if (rnorm < tol * rnorm_init) {
			converged = TRUE;
		}

	}

	CdamFreeDevice(F, num_node * sizeof(f64));
	CdamFreeDevice(dx, num_node * sizeof(f64));
	MatrixDestroy(J);
	CSRAttrDestroy(spy);
	KrylovDestroy(ksp);
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
		z = 1.5e-4 - coord[i * 3 + 2];
		if (z > eps) {
			h = 1.0;
		}
		else if (z < -eps) {
			h = 0.0;
		}
		else {
			h = 0.5 * (1.0 + z / eps + sin(M_PI * z / eps) / M_PI);
		}
		value[i] = h;
	}
}

int main() {
	b32 converged = FALSE;
	i32 step = 0, num_step = 10;
	char filename_buffer[256] = {0};
	struct cudaDeviceProp prop;
	i32 num_device;
	cudaGetDeviceCount(&num_device);
	ASSERT(num_device > 0 && "No CUDA device found");
	if (num_device) {
		cudaGetDeviceProperties(&prop, 0);
		printf("Device name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Total global memory: %f (MB)\n", (f64)prop.totalGlobalMem / 1024.0 / 1024.0);
		printf("Shared memory per block: %f (KB)\n", (f64)prop.sharedMemPerBlock / 1024.0);
		printf("Registers per block: %f (KB)\n", (f64)prop.regsPerBlock / 1024.0);
		printf("Warp size: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max threads dimension: %d X %d X %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid size: %d X %d X %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("Clock rate: %f (Hz) \n", (f64)prop.clockRate);
		printf("Total constant memory: %f (MB)\n", (f64)prop.totalConstMem / 1024.0 / 1024.0);
		printf("Peak memory clock rate: %f (MHz)\n", (f64)prop.memoryClockRate / 1e6);
		printf("Memory bandwidth: %f (GB/s)\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
	}

	H5FileInfo* h5_handler = H5OpenFile("cube.h5", "r");
	Mesh3D* mesh = Mesh3DCreateH5(h5_handler, "mesh");
	H5CloseFile(h5_handler);

	Mesh3DColor(mesh);
	return 0;


	Field* wgold = FieldCreate3D(mesh, 1);
	Field* dwgold = FieldCreate3D(mesh, 1);
	Field* dwg = FieldCreate3D(mesh, 1);
	FieldUpdateDevice(wgold);
	FieldUpdateDevice(dwgold);
	FieldUpdateDevice(dwg);

	ParticleContext* pctx = ParticleContextCreate(100);
	ParticleContextUpdateDevice(pctx);
	if (step) {
		sprintf(filename_buffer, "sol.%d.h5", step);
		h5_handler = H5OpenFile(filename_buffer, "r");
		FieldLoad(wgold, h5_handler, "w");
		FieldLoad(dwgold, h5_handler, "dw");
		ParticleContextLoad(pctx, h5_handler, "ptc/test/group/context");
		FieldCopy(dwg, dwgold);
		H5CloseFile(h5_handler);
	}
	else {
		/* Initial condition */
		FieldInit(wgold, MyFieldInit, mesh);
	}

	while(step++ < num_step) {
		/* Prediction stage */
		ArrayScale(FieldDevice(dwg), (kGAMMA - 1.0) / kGAMMA);

		/* Generate new particles */
		ParticleContextAdd(pctx);

		/* Newton-Raphson iteration */
		while(!converged) {
			SolveFlowSystem(mesh, wgold, dwgold, dwg);
#ifdef DEBUG
			SolveParticleSystem(pctx);
#endif
			converged = TRUE;
		}

		/* Update stage */
		ArrayAXPY(FieldDevice(wgold), kDT * (1.0 - kGAMMA), FieldDevice(dwgold));
		ArrayAXPY(FieldDevice(wgold), kDT * kGAMMA, FieldDevice(dwg));
		ArrayCopy(FieldDevice(dwgold), FieldDevice(dwg), D2D);

		/* Particle update */
		ParticleContextUpdate(pctx);
		/* Particle removal */
		ParticleContextRemove(pctx);

		if (step % 10 == 0) {
			sprintf(filename_buffer, "sol.%d.h5", step);
			h5_handler = H5OpenFile(filename_buffer, "w");
			FieldUpdateHost(wgold);
			FieldUpdateHost(dwgold);
			FieldSave(wgold, h5_handler, "w");
			FieldSave(dwgold, h5_handler, "dw");

			ParticleContextUpdateHost(pctx);
			ParticleContextSave(pctx, h5_handler, "ptc/test/group/context");
			H5CloseFile(h5_handler);
		}	
	}


	ParticleContextDestroy(pctx);
	FieldDestroy(wgold);
	FieldDestroy(dwgold);
	FieldDestroy(dwg);
	Mesh3DDestroy(mesh);
	return 0;
}
