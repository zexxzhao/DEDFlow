#ifndef __FORM_H__
#define __FORM_H__

#include "common.h"


__BEGIN_DECLS__
#define MAX_NUM_OPTIONS (64)

enum OptionName {
	OPTION_RHOC = 0,
	OPTION_ALPHA_M = 1,
	OPTION_ALPHA_F = 2,
	OPTION_GAMMA = 3,
	OPTION_DT = 4,
	OPTION_NS = 5,
	OPTION_NS_SUPG = 6,
	OPTION_NS_PSPG = 7,
	OPTION_NS_LSIC = 8,
	OPTION_NS_OTHER = 9,
	OPTION_NS_TAUBAR = 10,
	OPTION_NS_KDC = 11,
	OPTION_T = 12,
	OPTION_T_SUPG = 13,
	OPTION_T_KDC = 14,
	OPTION_PHI = 15,
	OPTION_PHI_SUPG = 16,
	OPTION_PHI_KDC = 17
};

typedef enum OptionName OptionName;
typedef value_type FEMOptions[MAX_NUM_OPTIONS];

__END_DECLS__


#endif /* __FORM_H__ */
