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


#include "json.h"
#include "alloc.h"
#include "h5util.h"
#include "Mesh.h"
#include "Field.h"
#include "Particle.h"
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

void AssembleSystem(Mesh3D* mesh,
										f64* wgalpha, f64* dwgalpha,
										f64* F, Matrix* J,
										Dirichlet** bcs, index_type nbc) {
	static int flag = 0;
	// index_type i, j, k;
	index_type num_node = (index_type)Mesh3DNumNode(mesh);
	index_type num_tet = (index_type)Mesh3DNumTet(mesh);
	index_type num_prism = (index_type)Mesh3DNumPrism(mesh);
	index_type num_hex = (index_type)Mesh3DDataNumHex(mesh);
	// f64* xg = Mesh3DDataCoord(Mesh3DHost(mesh));

	/* Zero the RHS and LHS */
	if(F) {
		cudaMemset(F, 0, num_node * SIZE_OF(f64) * BS);
	}
	if(J) {
		MatrixZero(J);
	}

	if(num_tet) {
		AssembleSystemTet(mesh, wgalpha, dwgalpha, F, J);
		AssembleSystemTetFace(mesh, wgalpha, dwgalpha, F, J);
	}
	CUGUARD(cudaGetLastError());
	
	if(num_prism) {
	}

	if(num_hex) {
	}

	if(F) {
		cudaMemset(F + 4 * num_node, 0, num_node * SIZE_OF(f64) * 2);
		CUGUARD(cudaGetLastError());
	}
	for(index_type ibc = 0; ibc < nbc; ++ibc) {
		if(F) {
			DirichletApplyVec(bcs[ibc], F);
		}
		if(J) {
			DirichletApplyMat(bcs[ibc], J);
		}
	}
}

void SolveFlowSystem(Mesh3D* mesh,
										 f64* wgold, f64* dwgold, f64* dwg,
										 Matrix* J, f64* F, f64* dx,
										 Krylov* ksp,
										 Dirichlet** bcs, index_type nbc) {

#ifdef DBG_TET
	index_type maxit = 1;
#else
	index_type maxit = 4;
#endif
	index_type iter = 0;
	f64 tol = 0.5e-3;
	b32 converged = FALSE;
	f64 rnorm[4], rnorm_init[4];
	f64 minus_one = -1.0;
	clock_t start, end;

	f64 fact1[] = {1.0 - kALPHAM, kALPHAM, kALPHAM - 1.0, -kALPHAM};
	f64 fact2[] = {kDT * kALPHAF * (1.0 - kGAMMA), kDT * kALPHAF * kGAMMA,
								 -kDT * kALPHAF * (1.0 - kGAMMA), -kDT * kALPHAF * kGAMMA};

	cublasHandle_t handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE);

	index_type num_node = Mesh3DDataNumNode(Mesh3DHost(mesh));

	f64* wgalpha = (f64*)CdamMallocDevice(num_node * SIZE_OF(f64) * BS);
	f64* dwgalpha = (f64*)CdamMallocDevice(num_node * SIZE_OF(f64) * BS);


	/* dwgalpha = fact1[0] * dwgold + fact1[1] * dwg */ 
	cudaMemset(dwgalpha, 0, num_node * SIZE_OF(f64) * BS);
	cublasDaxpy(handle, num_node * BS, fact1 + 0, dwgold, 1, dwgalpha, 1);
	cublasDaxpy(handle, num_node * BS, fact1 + 1, dwg, 1, dwgalpha, 1);
	/* dwgalpha[3, :] = dwg[3, :] */
	cublasDcopy(handle, num_node, dwg + num_node * 3, 1, dwgalpha + num_node * 3, 1);

	/* wgalpha = wgold + fact2[0] * dwgold + fact2[1] * dwg */
	cublasDcopy(handle, num_node * BS, wgold, 1, wgalpha, 1);
	cublasDaxpy(handle, num_node * BS, fact2 + 0, dwgold, 1, wgalpha, 1);
	cublasDaxpy(handle, num_node * BS, fact2 + 1, dwg, 1, wgalpha, 1);
	cudaMemset(wgalpha + num_node * 3, 0, num_node * SIZE_OF(f64));


	/* Construct the right-hand side */
	
	start = clock();
	AssembleSystem(mesh, wgalpha, dwgalpha, F, NULL, bcs, nbc);
	end = clock();
	fprintf(stdout, "Assemble F time: %f ms\n", (f64)(end - start) / CLOCKS_PER_SEC * 1000.0);
	cublasDnrm2(handle, num_node * 3, F + num_node * 0, 1, rnorm_init + 0);
	cublasDnrm2(handle, num_node * 1, F + num_node * 3, 1, rnorm_init + 1);
	cublasDnrm2(handle, num_node * 1, F + num_node * 4, 1, rnorm_init + 2);
	cublasDnrm2(handle, num_node * 1, F + num_node * 5, 1, rnorm_init + 3);



	CUGUARD(cudaGetLastError());

	fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", 0, rnorm_init[0], 1.0, tol);
	fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", 0, rnorm_init[1], 1.0, tol);
	fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", 0, rnorm_init[2], 1.0, tol);
	fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", 0, rnorm_init[3], 1.0, tol);
	rnorm_init[0] += 1e-16;
	rnorm_init[1] += 1e-16;
	rnorm_init[2] += 1e-16;
	rnorm_init[3] += 1e-16;

	while(!converged && iter < maxit) {
		/* Construct the Jacobian matrix */
		start = clock();
		AssembleSystem(mesh, wgalpha, dwgalpha, NULL, J, bcs, nbc);
		end = clock();
		fprintf(stdout, "Assemble J time: %f ms\n", (f64)(end - start) / CLOCKS_PER_SEC * 1000.0);
		CUGUARD(cudaGetLastError());

	
		/* Solve the linear system */
		cudaMemset(dx, 0, num_node * SIZE_OF(f64) * BS);
		// for(index_type ibc = 0; ibc < nbc; ++ibc) {
		// 	DirichletApplyVec(bcs[ibc], dx);
		// }
		cudaStreamSynchronize(0);
		start = clock();
		KrylovSolve(ksp, J, dx, F);

		cudaStreamSynchronize(0);
		end = clock();
		fprintf(stdout, "Krylov solve time: %f ms\n", (f64)(end - start) / CLOCKS_PER_SEC * 1000.0);

		/* Update the solution */	
		// cublasDaxpy(handle, ArrayLen(d_dwg), &minus_one, dx, 1, ArrayData(d_dwg), 1);
		/* Update dwg by dwg -= dx */
		cublasDaxpy(handle, num_node * BS, &minus_one, dx, 1, dwg, 1);
		// cublasDaxpy(handle, num_node * 3, &minus_one, dx + num_node * 0, 1, dwg + num_node * 0, 1);
		// cublasDaxpy(handle, num_node * 1, &minus_one, dx + num_node * 3, 1, dwg + num_node * 3, 1);
		if(0) {
			/* Update dwgalpha by dwgalpha[[0,1,2], :] -= kALPHAM * dx[[0,1,2], :] */ 
			cublasDaxpy(handle, num_node * 3, fact1 + 3, dx + num_node * 0, 1, dwgalpha + num_node * 0, 1);
			/* Update dwgalpha by dwgalpha[3, :] -= dx[3, :] */ 
			cublasDaxpy(handle, num_node * 1, &minus_one, dx + num_node * 3, 1, dwgalpha + num_node * 3, 1);
			/* Update dwgalpha by dwgalpha[[4,5], :] -= kALPHAM * dx[[4,5], :] */ 
			cublasDaxpy(handle, num_node * 2, fact1 + 3, dx + num_node * 4, 1, dwgalpha + num_node * 4, 1);
			/* Update wgalpha by wgalpha[[0,1,2], :] -= kDT * kALPHAF * kGAMMA * dx[[0,1,2], :] */
			cublasDaxpy(handle, num_node * 3, fact2 + 3, dx + num_node * 0, 1, wgalpha + num_node * 0, 1);
			/* Update wgalpha by wgalpha[[4,5], :] -= kDT * kALPHAF * kGAMMA * dx[[4,5], :] */
			cublasDaxpy(handle, num_node * 2, fact2 + 3, dx + num_node * 4, 1, wgalpha + num_node * 4, 1);
		}
		else {
			/* dwgalpha = fact1[0] * dwgold + fact1[1] * dwg */ 
			cudaMemset(dwgalpha, 0, num_node * SIZE_OF(f64) * BS);
			cublasDaxpy(handle, num_node * BS, fact1 + 0, dwgold, 1, dwgalpha, 1);
			cublasDaxpy(handle, num_node * BS, fact1 + 1, dwg, 1, dwgalpha, 1);
			/* dwgalpha[3, :] = dwg[3, :] */
			cublasDcopy(handle, num_node * 1, dwg + num_node * 3, 1, dwgalpha + num_node * 3, 1);

			/* wgalpha = wgold + fact2[0] * dwgold + fact2[1] * dwg */
			cublasDcopy(handle, num_node * BS, wgold, 1, wgalpha, 1);
			cublasDaxpy(handle, num_node * BS, fact2 + 0, dwgold, 1, wgalpha, 1);
			cublasDaxpy(handle, num_node * BS, fact2 + 1, dwg, 1, wgalpha, 1);
			cudaMemset(wgalpha + num_node * 3, 0, num_node * SIZE_OF(f64));

		}


		start = clock();
		AssembleSystem(mesh, wgalpha, dwgalpha, F, NULL, bcs, nbc);
		end = clock();
		fprintf(stdout, "Assemble F time: %f ms\n", (f64)(end - start) / CLOCKS_PER_SEC * 1000.0);
		cublasDnrm2(handle, num_node * 3, F + num_node * 0, 1, rnorm + 0);
		cublasDnrm2(handle, num_node * 1, F + num_node * 3, 1, rnorm + 1);	
		cublasDnrm2(handle, num_node * 1, F + num_node * 4, 1, rnorm + 2);
		cublasDnrm2(handle, num_node * 1, F + num_node * 5, 1, rnorm + 3);
		fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", iter + 1, rnorm[0], rnorm[0] / rnorm_init[0], tol);
		fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", iter + 1, rnorm[1], rnorm[1] / rnorm_init[1], tol);
		fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", iter + 1, rnorm[2], rnorm[2] / rnorm_init[2], tol);
		fprintf(stdout, "Newton %d) abs = %.17e rel = %6.4e (tol = %6.4e)\n", iter + 1, rnorm[3], rnorm[3] / rnorm_init[3], tol);

		if (rnorm[0] < tol * rnorm_init[0] &&
				rnorm[1] < tol * rnorm_init[1] &&
				rnorm[2] < tol * rnorm_init[2] &&
				rnorm[3] < tol * rnorm_init[3]) {
			converged = TRUE;
		}
		iter++;

	}

	CdamFreeDevice(dwgalpha, num_node * SIZE_OF(f64) * BS);
	CdamFreeDevice(wgalpha, num_node * SIZE_OF(f64) * BS);
}


void MyFieldInit(f64* value, void* ctx) {
	double eps = 1.5e-4 * 0.5;
	double z, h;
	Mesh3D* mesh = (Mesh3D*)ctx;
	Mesh3DData* data = Mesh3DHost(mesh);
	index_type i;
	index_type num_node = Mesh3DDataNumNode(data);
	f64* coord = Mesh3DDataCoord(data);

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
		value[num_node * 4 + i] = coord[i * 3 + 0];
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
	CDAM_Mesh* mesh;
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
	fprintf(stdout, json_str);
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
	}

	H5FileInfo* h5_handler = H5OpenFile(JSONGetItem(config, "IO.Input.Path")->valuestring, "r");
	CDAM_MeshCreate(MPI_COMM_WORLD, &mesh);
	CDAM_MeshLoad(mesh, h5_handler);
	H5CloseFile(h5_handler);

	CDAM_MeshToDevice(mesh);
	CDAM_MeshGenreateColorBatch(mesh);

	cublasHandle_t handle = *(cublasHandle_t*)GlobalContextGet(GLOBAL_CONTEXT_CUBLAS_HANDLE);
	// cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
	cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
	
	// cusparseHandle_t sp_handle;
	// cusparseCreate(&sp_handle);

	CDAM_Form vms;
	CDAM_FormCreate(MPI_COMM_WORLD, &vms);
	CDAM_FormSetMesh(vms, mesh);
	CDAM_FormConfig(vms, JSONGetItem(config, "VMS"));
	
	CDAM_NewtonSolver* nssolver;
	/* Create the Newton solver */
	CDAM_NewtonSolverCreate(MPI_COMM_WORLD, &nssolver);
	/* Set the FEM problem to be solved. This will be used to preallocated the distributed
	 * vectors and matrices. */
	CDAM_NewtonSolverSetFEM(nssolver, vms);
	/* Set the number of Newton iterations, 
	 *     the relative residual 
	 *     the absolute residual
	 *     whether use user-defined UpdateSolution */
	CDAM_NewtonSolverConfig(nssolver, JSONGetItem(config, "NewtonSolver"));

	CDAM_Krylov* krylov = NULL;
	CDAM_NewtonSolverGetLinearSolver(nssolver, &krylov);
	
	CDAM_KrylovSetUp(krylov, mesh, JSONGetItem(config, "NewtonSolver.LinearSolver"));

	CDAM_Vec* vec, *dx;
	CDAM_Mat* mat;
	CDAM_NewtonSolverGetLinearSystem(nssolver, &mat, &dx, &vec);
	CDAM_VecCreate(MPI_COMM_WORLD, &vec);
	CDAM_VecSetUp(vec, vms);
	CDAM_VecCreate(MPI_COMM_WORLD, &dx);
	CDAM_VecClone(dx, vec);

	CDAM_MatCreate(MPI_COMM_WOTLD, &mat);
	CDAM_MatSetUp(mat, vms);


	/* Simulation initialization */
	value_type* wgold = (value_type*)CdamMallocDevice(num_node * SIZE_OF(value_type) * BS);
	value_type* dwgold = (value_type*)CdamMallocDevice(num_node * SIZE_OF(value_type) * BS);
	value_type* dwg = (value_type*)CdamMallocDevice(num_node * SIZE_OF(value_type) * BS);

	value_type* buffer = (value_type*)CdamMallocHost(num_node * SIZE_OF(value_type) * BS);

	cudaMemset(wgold, 0, num_node * SIZE_OF(value_type) * BS);
	cudaMemset(dwgold, 0, num_node * SIZE_OF(value_type) * BS);
	cudaMemset(dwg, 0, num_node * SIZE_OF(value_type) * BS);

	memset(buffer, 0, num_node * SIZE_OF(value_type) * BS);

	if(step) {
		sprintf(filename_buffer, "sol.%d.h5", step);
		h5_handler = H5OpenFile(filename_buffer, "r");
		H5ReadDatasetf64(h5_handler, "u", buffer);
		H5ReadDatasetf64(h5_handler, "phi", buffer + num_node * 4);
		H5ReadDatasetf64(h5_handler, "T", buffer + num_node * 5);
		cudaMemcpy(wgold, buffer, num_node * SIZE_OF(f64) * 3, cudaMemcpyHostToDevice);

		memset(buffer, 0, num_node * SIZE_OF(f64) * BS);
		H5ReadDatasetf64(h5_handler, "du", buffer);
		H5ReadDatasetf64(h5_handler, "p", buffer + num_node * 3);
		H5ReadDatasetf64(h5_handler, "phi", buffer + num_node * 4);
		H5ReadDatasetf64(h5_handler, "T", buffer + num_node * 5);
		cudaMemcpy(dwgold, buffer, num_node * SIZE_OF(f64) * BS, cudaMemcpyHostToDevice);

		// ParticleContextLoad(pctx, h5_handler, "ptc/test/group/context");
		
		cudaMemcpy(dwg, dwgold, num_node * SIZE_OF(f64) * BS, cudaMemcpyDeviceToDevice);

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
		cudaMemset(dwgold, 0, num_node * SIZE_OF(f64) * BS);
		cudaMemset(dwg, 0, num_node * SIZE_OF(f64) * BS);
		cudaMemset(wgold, 0, num_node * SIZE_OF(f64) * BS);
		cudaMemcpy(wgold, buffer, num_node * SIZE_OF(f64) * BS, cudaMemcpyHostToDevice);
		cudaMemset(wgold + num_node * 3, 0, num_node * SIZE_OF(f64));
		// cudaMemcpy(dwgold + num_node * 3, buffer + num_node * 3, num_node * SIZE_OF(f64), cudaMemcpyHostToDevice);
		cudaMemcpy(dwg + num_node * 3, buffer + num_node * 3, num_node * SIZE_OF(f64), cudaMemcpyHostToDevice);
	
		h5_handler = H5OpenFile("sol.0.h5", "w");
		H5WriteDatasetf64(h5_handler, "u", num_node * 3, buffer);
		H5WriteDatasetf64(h5_handler, "p", num_node, buffer + num_node * 3);
		H5WriteDatasetf64(h5_handler, "phi", num_node, buffer + num_node * 4);
		H5WriteDatasetf64(h5_handler, "T", num_node, buffer + num_node * 5);

		memset(buffer, 0, num_node * SIZE_OF(f64) * BS);
		H5WriteDatasetf64(h5_handler, "du", num_node * 3, buffer);
		H5WriteDatasetf64(h5_handler, "dphi", num_node, buffer + num_node * 4);
		H5WriteDatasetf64(h5_handler, "dT", num_node, buffer + num_node * 5);

		H5CloseFile(h5_handler);

	}

	/* Simulation loop */
	f64 fac_pred[] = {(kGAMMA - 1.0) / kGAMMA};
	f64 fac_corr[] = {kDT * (1.0 - kGAMMA), kDT * kGAMMA};

	while(step ++ < num_step) {
		fprintf(stdout, "##################\n");
		fprintf(stdout, "# Step %d\n", step);
		fprintf(stdout, "##################\n");
		fflush(stdout);
		/* Prediction stage */
		// ArrayScale(FieldDevice(dwg), (kGAMMA - 1.0) / kGAMMA);
		cublasDscal(handle, num_node * 3, fac_pred, dwg, 1);
		cublasDscal(handle, num_node * 2, fac_pred, dwg + num_node * 4, 1);

		/* Generate new particles */
		// ParticleContextAdd(pctx);

		/* Newton-Raphson iteration */
		converged = FALSE;
		while(!converged) {
			CDAM_NewtonSolverSolve(nssolver, wgold, dwgold, dwg, NULL, NULL);
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

		if (step % 10 == 0) {
			sprintf(filename_buffer, "sol.%d.h5", step);
			fprintf(stdout, "Save solution to %s\n", filename_buffer);
			h5_handler = H5OpenFile(filename_buffer, "w");
		
			cudaMemcpy(buffer, wgold, num_node * SIZE_OF(f64) * BS, cudaMemcpyDeviceToHost);
			cudaMemcpy(buffer + num_node * 3, dwgold + num_node * 3, num_node * SIZE_OF(f64), cudaMemcpyDeviceToHost);
			H5WriteDatasetf64(h5_handler, "u", num_node * 3, buffer);
			H5WriteDatasetf64(h5_handler, "phi", num_node, buffer + num_node * 4);
			H5WriteDatasetf64(h5_handler, "T", num_node, buffer + num_node * 5);

			cudaMemcpy(buffer, dwgold, num_node * SIZE_OF(f64) * BS, cudaMemcpyDeviceToHost);
			H5WriteDatasetf64(h5_handler, "du", num_node * 3, buffer);
			H5WriteDatasetf64(h5_handler, "p", num_node, buffer + num_node * 3);
			H5WriteDatasetf64(h5_handler, "dphi", num_node, buffer + num_node * 4);
			H5WriteDatasetf64(h5_handler, "dT", num_node, buffer + num_node * 5);
			
			// ParticleContextUpdateHost(pctx);
			// ParticleContextSave(pctx, h5_handler, "ptc/test/group/context");
			H5CloseFile(h5_handler);
		}	
	}


	CDAM_NewtonSolverGetLinearSystem(nssolver, &mat, &dx, &vec);
	CDAM_VecDestroy(vec);
	CDAM_VecDestroy(dx);
	CDAM_MatDestroy(mat);

	CDAM_NewtonSolverGetLinearSolver(nssolver, &krylov);
	CDAM_KrylovDestroy(krylov);

	CDAM_FormDestroy(vms);
	CDAM_NewtonSolverDestroy(nssolver);
	CDAM_MeshDestroy(mesh);



	Finalize();
	return 0;
}
