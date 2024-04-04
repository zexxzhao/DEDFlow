#include "common.h"

__BEGIN_DECLS__

static __global__ void
GMRESUpdateResidualUpdateKernel(f64* beta, f64* gv) {
	f64 beta_new[2];
	beta_new[0] = beta[0];
	/* cos = gv[0], sin = gv[1] */
	beta_new[1] = -gv[1] * beta_new[0];
	beta_new[0] *= gv[0];
	beta[0] = beta_new[0];
	beta[1] = beta_new[1];
}


void GMRESResidualUpdatePrivate(f64* beta, f64* gv) {
	GMRESUpdateResidualUpdateKernel<<<1, 1>>>(beta, gv);
}


__END_DECLS__
