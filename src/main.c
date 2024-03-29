#include <stdio.h>
#include <stdlib.h>

#include "h5util.h"
#include "Mesh.h"
#include "Field.h"
#include "Particle.h"

#define kRHOC (0.5)
#define kDT (0.1)
#define kALPHAM ((3.0 - kRHOC) / (1.0 + kRHOC))
#define kALPHAF (1.0 / (1.0 + kRHOC))
#define kGAMMA (0.5 + kALPHAM - kALPHAF)


int main() {
	b32 converged = FALSE;
	char filename_buffer[256];
	H5FileInfo* h5_handler = H5OpenFile("example_tet_mesh.h5", "r");
	Mesh3D* mesh = Mesh3DCreateH5(h5_handler, "mesh");
	H5CloseFile(h5_handler);

	i32 num_nodal_dof = 6;
	Field* wgold = FieldCreate3D(mesh, num_nodal_dof);
	Field* dwgold = FieldCreate3D(mesh, num_nodal_dof);
	Field* dwg = FieldCreate3D(mesh, num_nodal_dof);

	ParticleContext* pctx = ParticleContextCreate(100);

	i32 step = 0, num_step = 10;
	if (step == 0) {
		// FieldInit(wgold, wgoldInit);
		// FieldInit(dwgold, dwgoldInit);
		// FieldInit(dwg, dwgInit);
		// ParticleInit(pctx);
	}
	else {
		sprintf(filename_buffer, "sol.%d.h5", step);
		h5_handler = H5OpenFile(filename_buffer, "r");
		FieldLoad(wgold, h5_handler, "w");
		FieldLoad(dwgold, h5_handler, "dw");
		FieldCopy(dwg, dwgold);
		ParticleContextLoad(pctx, h5_handler, "ptc");
	}
	FieldUpdateDevice(wgold);
	FieldUpdateDevice(dwgold);
	FieldUpdateDevice(dwg);
	ParticleContextUpdateDevice(pctx);

	while(step++ < num_step) {
		/* Prediction stage */
		ArrayScale(FieldDevice(dwg), (kGAMMA - 1.0) / kGAMMA);

		/* Generate new particles */
		ParticleContextAdd(pctx);

		/* Newton-Raphson iteration */
		while(!converged) {
#ifdef DEBUG
			SolveFlowSystem(wgold, dwgold, dwg);
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
