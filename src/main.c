#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[]) {
	Mesh* mesh;
	MeshCrate(&mesh);
	MeshLoad(mesh, "mesh/file/name");
	MeshMove(mesh, HostToDevice);

	Field* wgold, *dwgold, *dwg;
	i32 nDofPerNode = 6;
	FieldCreate(&wgold, mesh, nDofPerNode);
	FieldCreate(&dwgold, mesh, nDofPerNode);
	FieldCreate(&dwg, mesh, nDofPerNode);

	ParticleContext* pctx;
	ParticleContextCreate(&pctx);

	i32 step = 0;
	if (step == 0) {
		FieldInit(wgold, wgoldInit);
		FieldInit(dwgold, dwgoldInit);
		FieldInit(dwg, dwgInit);
		ParticleInit(pctx);
	}
	else {
		FieldLoad(wgold, "path/to/wgold");
		FieldLoad(dwgold, "path/to/dwgold");
		FieldCopy(dwg, dwgold);
		ParticleLoad(pctx, "path/to/pctx");
	}
	FieldMove(wgold, HostToDevice);
	FieldMove(dwgold, HostToDevice);
	FieldMove(dwg, HostToDevice);
	ParticleMove(pctx, HostToDevice);



	while(step++ < nStep) {
		/* Prediction stage */
		ArrayScale(FieldGetArray(dwg), FieldGetArrayLen(dwg), (kGAMMA - 1.0) / kGAMMA);

		/* Generate new particles */
		ParticleGenerate(pctx);

		/* Newton-Raphson iteration */
		while(not converged) {
			SolveFlowSystem(wgold, dwgold, dwg);
			SolveParticleSystem(pctx);
		}

		/* Update stage */
		ArrayAXPY(FielGetArray(wgold), kDT * (1.0 - kGAMMA), FieldGetArray(dwgold), FieldGetArrayLen(wgold));
		ArrayAXPY(FielGetArray(wgold), kDT * kGAMMA, FieldGetArray(dwg), FieldGetArrayLen(wgold));
		ArrayCopy(FieldGetArray(dwgold), FieldGetArray(dwg), FieldGetArrayLen(dwg));

		/* Particle update */
		ParticleUpdate(pctx);
		/* Particle removal */
		ParticleRemove(pctx);

		if (step % 10 == 0) {
			FieldMove(wgold, DeviceToHost);
			FieldSave(wgold, "path/to/wgold");
			FieldMove(dwgold, HostToDevice);
			FieldSave(dwgold, "path/to/dwgold");

			ParticleMove(pctx, DeviceToHost);
			ParticleSave(pctx, "path/to/pctx");
		}	
	}



	ParticleContextDestroy(pctx);
	FieldDestroy(wgold);
	FieldDestroy(dwgold);
	FieldDestroy(dwg);
	MeshDestroy(mesh);
	return 0;
}
