#ifndef __PARTICLE_H__
#define __PARTICLE_H__

#include <stdlib.h>
#include <string.h>
#include "Array.h"

__BEGIN_DECLS__

typedef struct H5FileInfo H5FileInfo;

typedef struct ParticleContext ParticleContext;
struct ParticleContext {
	index_type num_particle;
	i32 num_pointwise_dof; /* number of pointwise degrees of freedom (=9) */
	Array* h_arr[3]; /* host array */
	Array* d_arr[3]; /* device array */
	f64 buff[2]; /* context */
};


ParticleContext* ParticleContextCreate(index_type num_particle);
void ParticleContextDestroy(ParticleContext* ctx);

void ParticleContextCopy(ParticleContext* dst, const ParticleContext* src);

void ParticleContextLoad(ParticleContext* ctx, H5FileInfo* h5f, const char* group_name);
void ParticleContextSave(const ParticleContext* ctx, H5FileInfo* h5f, const char* group_name);

void ParticleContextUpdateHost(ParticleContext* ctx);
void ParticleContextUpdateDevice(ParticleContext* ctx);

void ParticleContextAdd(ParticleContext* ctx);
void ParticleContextUpdate(ParticleContext* ctx);
void ParticleContextRemove(ParticleContext* ctx);

#define ParticleCTXHost(pctx) ((pctx)->host)
#define ParticleCTXDevice(pctx) ((pctx)->device)
#define ParticleCTXNumParticle(pctx) ((pctx)->num_particle)

#define ParticleCTXHostCoord(pctx) ((pctx)->h_arr[0])
#define ParticleCTXHostVel(pctx) ((pctx)->h_arr[1])
#define ParticleCTXHostAcc(pctx) ((pctx)->h_arr[2])

#define ParticleCTXDeviceCoord(pctx) ((pctx)->d_arr[0])
#define ParticleCTXDeviceVel(pctx) ((pctx)->d_arr[1])
#define ParticleCTXDeviceAcc(pctx) ((pctx)->d_arr[2])

#define ParticleMass(pctx) (((pctx)->buff)[0])
#define ParticleRadius(pctx) (((pctx)->buff)[1])

__END_DECLS__

#endif /* __PARTICLE_H__ */
