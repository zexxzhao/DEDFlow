
#include "alloc.h"
#include "Particle.h"

__BEGIN_DECLS__
int sprintf(char *str, const char *format, ...);

ParticleContext* ParticleContextCreate(u32 num_particle) {
	ParticleContext* ctx;
	ctx = (ParticleContext*)CdamMallocHost(sizeof(ParticleContext));
	ParticleCTXNumParticle(ctx) = num_particle;
	ctx->num_pointwise_dof = 9;

	ParticleCTXHostCoord(ctx) = ArrayCreateHost(num_particle * 3);
	ParticleCTXHostVel(ctx) = ArrayCreateHost(num_particle * 3);
	ParticleCTXHostAcc(ctx) = ArrayCreateHost(num_particle * 3);

	ParticleCTXDeviceCoord(ctx) = ArrayCreateDevice(num_particle * 3);
	ParticleCTXDeviceVel(ctx) = ArrayCreateDevice(num_particle * 3);
	ParticleCTXDeviceAcc(ctx) = ArrayCreateDevice(num_particle * 3);

	/* The following values are hard coded for now */
	ParticleMass(ctx) = 1.0;
	ParticleRadius(ctx) = 0.1;
	/* Hard coded values end here */
	return ctx;
}

void ParticleContextDestroy(ParticleContext* ctx) {
	ArrayDestroy(ParticleCTXHostCoord(ctx));
	ArrayDestroy(ParticleCTXHostVel(ctx));
	ArrayDestroy(ParticleCTXHostAcc(ctx));
	ParticleCTXHostCoord(ctx) = NULL;
	ParticleCTXHostVel(ctx) = NULL;
	ParticleCTXHostAcc(ctx) = NULL;

	ArrayDestroy(ParticleCTXDeviceCoord(ctx));
	ArrayDestroy(ParticleCTXDeviceVel(ctx));
	ArrayDestroy(ParticleCTXDeviceAcc(ctx));
	ParticleCTXDeviceCoord(ctx) = NULL;
	ParticleCTXDeviceVel(ctx) = NULL;
	ParticleCTXDeviceAcc(ctx) = NULL;

	CdamFreeHost(ctx, sizeof(ParticleContext));
	ctx = NULL;
}

void ParticleContextCopy(ParticleContext* dst, const ParticleContext* src) {
	ASSERT(dst && "ParticleContextCopy: dst is NULL!");
	ASSERT(src && "ParticleContextCopy: src is NULL!");

	ASSERT(ParticleCTXNumParticle(dst) == ParticleCTXNumParticle(src) && "ParticleContextCopy: Number of particles do not match!");
	ASSERT(ParticleMass(dst) == ParticleMass(src) && "ParticleContextCopy: Mass do not match!");
	ASSERT(ParticleRadius(dst) == ParticleRadius(src) && "ParticleContextCopy: Radius do not match!");


	ArrayCopy(ParticleCTXHostCoord(dst), ParticleCTXHostCoord(src), H2H);
	ArrayCopy(ParticleCTXHostVel(dst), ParticleCTXHostVel(src), H2H);
	ArrayCopy(ParticleCTXHostAcc(dst), ParticleCTXHostAcc(src), H2H);

	ArrayCopy(ParticleCTXDeviceCoord(dst), ParticleCTXDeviceCoord(src), D2D);
	ArrayCopy(ParticleCTXDeviceVel(dst), ParticleCTXDeviceVel(src), D2D);
	ArrayCopy(ParticleCTXDeviceAcc(dst), ParticleCTXDeviceAcc(src), D2D);
}

void ParticleContextLoad(ParticleContext* ctx, H5FileInfo* h5f, const char* group_name) {
	char path[256];

	ASSERT(ctx && "ParticleContextLoad: ctx is NULL!");
	ASSERT(group_name && "ParticleContextLoad: group_name is NULL!");

	ASSERT(strlen(group_name) < 192 && "ParticleContextLoad: group_name is too long!");
	

	sprintf(path, "%s/coord", group_name); 
	ArrayLoad(ParticleCTXHostCoord(ctx), h5f, path);
	ArrayCopy(ParticleCTXDeviceCoord(ctx), ParticleCTXHostCoord(ctx), H2D);

	sprintf(path, "%s/vel", group_name);
	ArrayLoad(ParticleCTXHostVel(ctx), h5f, path);
	ArrayCopy(ParticleCTXDeviceVel(ctx), ParticleCTXHostVel(ctx), H2D);

	sprintf(path, "%s/acc", group_name);
	ArrayLoad(ParticleCTXHostAcc(ctx), h5f, path);
	ArrayCopy(ParticleCTXDeviceAcc(ctx), ParticleCTXHostAcc(ctx), H2D);
}

void ParticleContextSave(const ParticleContext* ctx, H5FileInfo* h5f, const char* group_name) {
	char path[256];
	ASSERT(ctx && "ParticleContextSave: ctx is NULL!");
	ASSERT(group_name && "ParticleContextSave: group_name is NULL!");

	ASSERT(strlen(group_name) < 192 && "ParticleContextSave: group_name is too long!");

	sprintf(path, "%s/coord", group_name);
	ArraySave(ParticleCTXHostCoord(ctx), h5f, path);

	sprintf(path, "%s/vel", group_name);
	ArraySave(ParticleCTXHostVel(ctx), h5f, path);

	sprintf(path, "%s/acc", group_name);
	ArraySave(ParticleCTXHostAcc(ctx), h5f, path);
}

void ParticleContextUpdateHost(ParticleContext* ctx) {
	ASSERT(ctx != NULL && "ParticleContext is not allocated!");
	ArrayCopy(ParticleCTXHostCoord(ctx), ParticleCTXDeviceCoord(ctx), D2H);
	ArrayCopy(ParticleCTXHostVel(ctx), ParticleCTXDeviceVel(ctx), D2H);
	ArrayCopy(ParticleCTXHostAcc(ctx), ParticleCTXDeviceAcc(ctx), D2H);
}


void ParticleContextUpdateDevice(ParticleContext* ctx) {
	ASSERT(ctx != NULL && "ParticleContext is not allocated!");
	ArrayCopy(ParticleCTXDeviceCoord(ctx), ParticleCTXHostCoord(ctx), H2D);
	ArrayCopy(ParticleCTXDeviceVel(ctx), ParticleCTXHostVel(ctx), H2D);
	ArrayCopy(ParticleCTXDeviceAcc(ctx), ParticleCTXHostAcc(ctx), H2D);
}

void ParticleContextAdd(ParticleContext* ctx) {
	UNUSED(ctx);
}

void ParticleContextUpdate(ParticleContext* ctx) {
	UNUSED(ctx);
}

void ParticleContextRemove(ParticleContext* ctx) {
	UNUSED(ctx);
}

__END_DECLS__
