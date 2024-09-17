#include <stdio.h>
#include <time.h>
#include <math.h>
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif
#include <stdlib.h>

// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <cuda_profiler_api.h>


#include "json.h"
#include "alloc.h"
#include "h5util.h"
#include "Mesh.h"
#include "vec.h"
// #include "Particle.h"
#include "NewtonSolver.h"
#include "dirichlet.h"
#include "csr.h"
#include "krylov.h"
#include "assemble.h"

#define kRHOC (0.5)
#define kDT (5e-2)
#define kALPHAM (0.5 * (3.0 - kRHOC) / (1.0 + kRHOC))
#define kALPHAF (1.0 / (1.0 + kRHOC))
#define kGAMMA (0.5 + kALPHAM - kALPHAF)

#define BS (6)




void MyFieldInit(value_type* value, void* ctx) {
	double eps = 1.5e-4 * 0.5;
	double z, h;
	CdamMesh* mesh = (CdamMesh*)ctx;
	index_type i;
	index_type num_node = CdamMeshNumNode(mesh);
	value_type* coord = CdamMeshCoord(mesh);

	for(i = 0; i < num_node; i++) {
#ifdef DBG_TET
		value[i * 3 + 0] = coord[i * 3 + 0];
		value[i * 3 + 1] = coord[i * 3 + 1];
		value[i * 3 + 2] = coord[i * 3 + 2];
#else
		value[i * 3 + 0] = 1.0; // coord[i * 3 + 0]; // * coord[i * 3 + 0] - 0.5;
		value[i * 3 + 1] = 0.0; // coord[i * 3 + 1];
		value[i * 3 + 2] = 0.0; // coord[i * 3 + 2];
#endif
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
		value[num_node * 5 + i] = -coord[i * 3 + 0];
	}
}

static inline void ParseJSONFile(const char* filename, cJSON** root) {
	FILE* fp = fopen(filename, "r");
	ASSERT(fp && "Cannot open file");
	fseek(fp, 0, SEEK_END);
	size_t size = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	char* buffer = (char*)CdamMallocHost(size + 1);
	fread(buffer, 1, size, fp);
	buffer[size] = '\0';
	fclose(fp);

	*root = cJSON_Parse(buffer);
	CdamFreeHost(buffer, size + 1);

	ASSERT(*root && "Cannot parse JSON file");

}

int main(int argc, char** argv) {
	int rank, num_procs;
	char argv_opt[256];
	CdamMesh* mesh;
	cJSON* config = NULL;
	{
		memset(argv_opt, 0, 256);
		for(i32 i = 1; i < argc;) {
			if(strcmp(argv[i], "--m") == 0) {
				strcpy(argv_opt, argv[i + 1]);
				i += 2;
			}
			else {
				i++;
			}
		}
	}
	if(strlen(argv_opt) == 0) {
		fprintf(stderr, "Usage: %s --m <config.json>\n", argv[0]);
		return 1;
	}

	
	ParseJSONFile(argv_opt, &config);


	char* json_str = cJSON_Print(config);
	fprintf(stdout, "%s", json_str);
	fprintf(stdout, "\n");
	// cJSON_Delete(config);
	CdamFreeHost(json_str, strlen(json_str) + 1);
	// return 0;

	Init(0, NULL);
	b32 converged = FALSE;
	i32 step = 0;
#ifdef DBG_TET
	i32 num_step = 1;
#else
	i32 num_step = 4000;
#endif
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	char filename_buffer[256] = {0};
	struct cudaDeviceProp prop;
	i32 num_device;
	cudaGetDeviceCount(&num_device);
	ASSERT(num_device == num_procs && "No CUDA device found");
	cudaSetDevice(rank);
	for(i32 i = 0; i < num_device; i++) {
		MPI_Barrier(MPI_COMM_WORLD);
		if(i == rank) {
			cudaGetDeviceProperties(&prop, i);
			printf("==================================================\n");
			printf(" * Device[%d] name: %s\n", rank, prop.name);
			printf(" * Compute capability: %d.%d\n", prop.major, prop.minor);
			printf(" * Total global memory: %f (MB)\n", (value_type)prop.totalGlobalMem / 1024.0 / 1024.0);
			printf(" * Shared memory per block: %f (KB)\n", (value_type)prop.sharedMemPerBlock / 1024.0);
			printf(" * Registers per block: %f (KB)\n", (value_type)prop.regsPerBlock / 1024.0);
			printf(" * Warp size: %d\n", prop.warpSize);
			printf(" * Max threads per block: %d\n", prop.maxThreadsPerBlock);
			printf(" * Max threads dimension: %d X %d X %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
			printf(" * Max grid size: %d X %d X %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
			printf(" * Clock rate: %f (Hz) \n", (value_type)prop.clockRate);
			printf(" * Total constant memory: %f (MB)\n", (value_type)prop.totalConstMem / 1024.0 / 1024.0);
			printf(" * Peak memory clock rate: %f (MHz)\n", (value_type)prop.memoryClockRate / 1e6);
			printf(" * Memory bandwidth: %f (GB/s)\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
			printf("==================================================\n");
		}
	}

	H5FileInfo* h5_handler = H5OpenFile(JSONGetItem(config, "IO.Input.Path")->valuestring, "r");
	CdamMeshCreate(MPI_COMM_WORLD, &mesh);
	CdamMeshLoad(mesh, h5_handler, "/mesh");
	H5CloseFile(h5_handler);

	CdamMeshPrefetch(mesh);
	// CdamMeshGenreateColorBatch(mesh);

#if CDAM_USE_CUDA
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
	// cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
	cublasDestroy(handle);
#endif
	
	/* Create the Newton solver */
	CdamNewtonSolver* nssolver;
	CdamNewtonSolverCreate(MPI_COMM_WORLD, &nssolver);
	/* Set the number of Newton iterations, 
	 *     the relative residual 
	 *     the absolute residual
	 *     whether use user-defined UpdateSolution */
	CdamNewtonSolverConfig(nssolver, JSONGetItem(config, "NewtonSolver"));

	CdamKrylov* krylov = NULL;
	CdamNewtonSolverGetLinearSolver(nssolver, &krylov);
	
	CdamKrylovSetup(krylov, mesh, JSONGetItem(config, "NewtonSolver.LinearSolver"));

	CdamVecLayout* layout;
	CdamVecLayoutCreate(&layout, config);
	value_type* vec, *dx;
	// CdamNewtonSolverGetLinearSystem(nssolver, &mat, &dx, &vec);
	
	index_type num_node = CdamMeshNumNode(mesh);
	dx = CdamTMalloc(value_type, num_node * sizeof(value_type) * BS, DEVICE_MEM);
	vec = CdamTMalloc(value_type, num_node * sizeof(value_type) * BS, DEVICE_MEM);
	CdamNewtonSolverB(nssolver) = vec;
	CdamNewtonSolverX(nssolver) = dx;

	CdamMat* mat;
	CdamMatCreate(MPI_COMM_WORLD, &mat);
	CdamNewtonSolverA(nssolver) = mat;


	/* Simulation initialization */
	value_type* wgold = CdamTMalloc(value_type, num_node * sizeof(value_type) * BS, DEVICE_MEM);
	value_type* dwgold = CdamTMalloc(value_type, num_node * sizeof(value_type) * BS, DEVICE_MEM);
	value_type* dwg = CdamTMalloc(value_type, num_node * sizeof(value_type) * BS, DEVICE_MEM);

	value_type* buffer = CdamTMalloc(value_type, num_node * sizeof(value_type) * BS, HOST_MEM);

	CdamMemset(wgold, 0, num_node * sizeof(value_type) * BS, DEVICE_MEM);
	CdamMemset(dwgold, 0, num_node * sizeof(value_type) * BS, DEVICE_MEM);
	CdamMemset(dwg, 0, num_node * sizeof(value_type) * BS, DEVICE_MEM);

	CdamMemset(buffer, 0, num_node * sizeof(value_type) * BS, HOST_MEM);

	if(step) {
		sprintf(filename_buffer, "sol.%d.h5", step);
		h5_handler = H5OpenFile(filename_buffer, "r");
		H5ReadDatasetf64(h5_handler, "u", buffer);
		H5ReadDatasetf64(h5_handler, "phi", buffer + num_node * 4);
		H5ReadDatasetf64(h5_handler, "T", buffer + num_node * 5);
		CdamMemcpy(wgold, buffer, num_node * sizeof(value_type) * 3, DEVICE_MEM, HOST_MEM);

		memset(buffer, 0, num_node * sizeof(value_type) * BS);
		H5ReadDatasetf64(h5_handler, "du", buffer);
		H5ReadDatasetf64(h5_handler, "p", buffer + num_node * 3);
		H5ReadDatasetf64(h5_handler, "phi", buffer + num_node * 4);
		H5ReadDatasetf64(h5_handler, "T", buffer + num_node * 5);
		// cudaMemcpy(dwgold, buffer, num_node * sizeof(value_type) * BS, cudaMemcpyHostToDevice);
		CdamMemcpy(dwgold, buffer, num_node * sizeof(value_type) * 3, DEVICE_MEM, HOST_MEM);

		
		CdamMemcpy(dwg, dwgold, num_node * sizeof(value_type) * 3, DEVICE_MEM, DEVICE_MEM);

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
		// cudaMemset(dwgold, 0, num_node * sizeof(value_type) * BS);
		// cudaMemset(dwg, 0, num_node * sizeof(value_type) * BS);
		// cudaMemset(wgold, 0, num_node * sizeof(value_type) * BS);
		// cudaMemcpy(wgold, buffer, num_node * sizeof(value_type) * BS, cudaMemcpyHostToDevice);
		// cudaMemset(wgold + num_node * 3, 0, num_node * sizeof(value_type));
		// cudaMemcpy(dwgold + num_node * 3, buffer + num_node * 3, num_node * sizeof(value_type), cudaMemcpyHostToDevice);
		// cudaMemcpy(dwg + num_node * 3, buffer + num_node * 3, num_node * sizeof(value_type), cudaMemcpyHostToDevice);
		CdamMemset(dwgold, 0, num_node * sizeof(value_type) * BS, DEVICE_MEM);
		CdamMemset(dwg, 0, num_node * sizeof(value_type) * BS, DEVICE_MEM);
		CdamMemset(wgold, 0, num_node * sizeof(value_type) * BS, DEVICE_MEM);
		CdamMemcpy(wgold, buffer, num_node * sizeof(value_type) * BS, DEVICE_MEM, HOST_MEM);
		CdamMemset(wgold + num_node * 3, 0, num_node, DEVICE_MEM);
		CdamMemcpy(dwg + num_node * 3, buffer + num_node * 3, num_node, DEVICE_MEM, HOST_MEM);
	
		h5_handler = H5OpenFile("sol.0.h5", "w");
		H5WriteDatasetf64(h5_handler, "u", num_node * 3, buffer);
		H5WriteDatasetf64(h5_handler, "p", num_node, buffer + num_node * 3);
		H5WriteDatasetf64(h5_handler, "phi", num_node, buffer + num_node * 4);
		H5WriteDatasetf64(h5_handler, "T", num_node, buffer + num_node * 5);

		memset(buffer, 0, num_node * sizeof(value_type) * BS);
		H5WriteDatasetf64(h5_handler, "du", num_node * 3, buffer);
		H5WriteDatasetf64(h5_handler, "dphi", num_node, buffer + num_node * 4);
		H5WriteDatasetf64(h5_handler, "dT", num_node, buffer + num_node * 5);

		H5CloseFile(h5_handler);

	}

	/* Simulation loop */
	value_type fac_pred[] = {(kGAMMA - 1.0) / kGAMMA};
	value_type fac_corr[] = {kDT * (1.0 - kGAMMA), kDT * kGAMMA};

	while(step ++ < num_step) {
		fprintf(stdout, "##################\n");
		fprintf(stdout, "# Step %d\n", step);
		fprintf(stdout, "##################\n");
		fflush(stdout);
		/* Prediction stage */
		// cublasDscal(handle, num_node * 3, fac_pred, dwg, 1);
		// cublasDscal(handle, num_node * 2, fac_pred, dwg + num_node * 4, 1);
		BLAS_CALL(scal, num_node * 3, fac_pred + 0, dwgold, 1);
		BLAS_CALL(scal, num_node * 2, fac_pred + 0, dwgold + num_node * 4, 1);

		/* Generate new particles */
		// ParticleContextAdd(pctx);

		/* Newton-Raphson iteration */
		converged = FALSE;
		while(!converged) {
			CdamNewtonSolverSolve(nssolver, dwg);
#ifdef DEBUG
			SolveParticleSystem(pctx);
#endif
			converged = TRUE;
		}

		/* Update stage */
		// cublasDaxpy(handle, num_node * 3, fac_corr + 0, dwgold, 1, wgold, 1);
		// cublasDaxpy(handle, num_node * 2, fac_corr + 0, dwgold + num_node * 4, 1, wgold + num_node * 4, 1);
		// cublasDaxpy(handle, num_node * 3, fac_corr + 1, dwg, 1, wgold, 1);
		// cublasDaxpy(handle, num_node * 2, fac_corr + 1, dwg + num_node * 4, 1, wgold + num_node * 4, 1);
		// cublasDcopy(handle, num_node * 6, dwg, 1, dwgold, 1);

		BLAS_CALL(axpy, num_node * 3, fac_corr + 0, dwgold, 1, wgold, 1);
		BLAS_CALL(axpy, num_node * 2, fac_corr + 0, dwgold + num_node * 4, 1, wgold + num_node * 4, 1);
		BLAS_CALL(axpy, num_node * 3, fac_corr + 1, dwg, 1, wgold, 1);
		BLAS_CALL(axpy, num_node * 2, fac_corr + 1, dwg + num_node * 4, 1, wgold + num_node * 4, 1);
		BLAS_CALL(copy, num_node * 6, dwg, 1, dwgold, 1);

		/* Particle update */
		// ParticleContextUpdate(pctx);
		/* Particle removal */
		// ParticleContextRemove(pctx);

		if (step % 10 == 0) {
			sprintf(filename_buffer, "sol.%d.h5", step);
			fprintf(stdout, "Save solution to %s\n", filename_buffer);
			h5_handler = H5OpenFile(filename_buffer, "w");
		
			cudaMemcpy(buffer, wgold, num_node * sizeof(value_type) * BS, cudaMemcpyDeviceToHost);
			cudaMemcpy(buffer + num_node * 3, dwgold + num_node * 3, num_node * sizeof(value_type), cudaMemcpyDeviceToHost);
			H5WriteDatasetf64(h5_handler, "u", num_node * 3, buffer);
			H5WriteDatasetf64(h5_handler, "phi", num_node, buffer + num_node * 4);
			H5WriteDatasetf64(h5_handler, "T", num_node, buffer + num_node * 5);

			cudaMemcpy(buffer, dwgold, num_node * sizeof(value_type) * BS, cudaMemcpyDeviceToHost);
			H5WriteDatasetf64(h5_handler, "du", num_node * 3, buffer);
			H5WriteDatasetf64(h5_handler, "p", num_node, buffer + num_node * 3);
			H5WriteDatasetf64(h5_handler, "dphi", num_node, buffer + num_node * 4);
			H5WriteDatasetf64(h5_handler, "dT", num_node, buffer + num_node * 5);
			
			// ParticleContextUpdateHost(pctx);
			// ParticleContextSave(pctx, h5_handler, "ptc/test/group/context");
			H5CloseFile(h5_handler);
		}	
	}


	// CdamVecDestroy(vec);
	// CdamVecDestroy(dx);
	CdamFree(vec, num_node * sizeof(value_type) * BS, DEVICE_MEM);
	CdamFree(dx, num_node * sizeof(value_type) * BS, DEVICE_MEM);
	CdamMatDestroy(mat);

	CdamNewtonSolverGetLinearSolver(nssolver, &krylov);
	CdamKrylovDestroy(krylov);

	CdamNewtonSolverDestroy(nssolver);
	CdamMeshDestroy(mesh);


	Finalize();
	return 0;
}
