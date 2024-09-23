# Description: Configuration file for CDAM
DEBUG=1



CDAM_USE_CUDA=1
CDAM_USE_MKL=0
CDAM_USE_ACCELERATE=0


# These packages are required
# HDF5: https://www.hdfgroup.org/downloads/hdf5/
ifeq ($(origin HDF5_ROOT), undefined)
$(error HDF5_ROOT is not set)
endif

# METIS: http://glaros.dtc.umn.edu/gkhome/metis/metis/download
ifeq ($(origin METIS_DIR), undefined)
$(error METIS_DIR is not set)
endif


# These packages are optional
ifeq ($(CDAM_USE_CUDA), 1)
# CUDA: https://developer.nvidia.com/cuda-downloads
ifeq ($(origin CUDA_ROOT), undefined)
$(error CUDA_ROOT is not set, which is required for CDAM_USE_CUDA=ON)
else
CUDA_DIR=$(CUDA_ROOT)
endif

ifeq ($(origin AMGX_DIR), undefined)
$(warning AMGX_DIR is not set)
endif

endif

ifndef PREFIX
PREFIX=$(PWD)
endif

###############################################################################
# Compiler and flags
###############################################################################
ifdef DEBUG
	C_OPT=-O0 -g3 -fsanitize=undefined
	CU_OPT=-O0 -g -G
else
	COPT=-O3 -g -DNDEBUG
	CU_OPT=-O3 -g -DNDEBUG
endif

FLAGS= -fPIC -Wall -Wextra \
			-Wconversion -Wdouble-promotion \
			-Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion \
			-Wno-deprecated-declarations
DEFINES=-DUSE_I32_INDEX -DUSE_F64_VALUE # -DCDAM_USE_CUDA # -DDBG_TET
CC=mpicc
CFLAGS=-std=c99 $(FLAGS) $(DEFINES) $(C_OPT) # -fno-omit-frame-pointer

INC=
LIB=

DEFINES+= -DUSE_HDF5
INC+=-I$(HDF5_ROOT)/include -I$(HYPRE_DIR)/include
LIB+=-L$(HDF5_ROOT)/lib -lhdf5 -L$(HYPRE_DIR)/lib -lHYPRE

DEFINES+= -DUSE_METIS
INC+=-I$(METIS_DIR)/include
LIB+=-L$(METIS_DIR)/lib -lmetis

ifeq ($(CDAM_USE_CUDA), 1)
DEFINES+= -DCDAM_USE_CUDA
INC+=-I$(CUDA_DIR)/include
LIB+=-lm
INC+=-I$(BLAS_DIR)/include -I$(LAPACK_DIR)/include
LIB+=-L$(BLAS_DIR)/lib -lblas -L$(LAPACK_DIR)/lib -llapack

ifneq ($(AMGX_DIR),)
DEFINES+= -DUSE_AMGX
INC+=-I$(AMGX_DIR)/include
LIB+=-L$(AMGX_DIR)/lib -lamgx -lcusolver -lnvToolsExt
endif

else ifeq ($(CDAM_USE_MKL), 1)
DEFINES+= -DCDAM_USE_MKL
INC+=-I$(MKLROOT)/include
LIB+=-L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
else ifeq ($(CDAM_USE_ACCELERATE), 1)
DEFINES+= -DCDAM_USE_ACCELERATE
LIB+=-framework Accelerate
endif



ifeq ($(CDAM_USE_CUDA), 1)
NVCC=nvcc
ifndef CU_ARCH # if not set, use the default value
CU_ARCH=$(shell __nvcc_device_query)
CU_ARCH:=$(strip $(CU_ARCH))
endif

# CU_ARCH=75: RTX 2080 Ti
# CU_ARCH=80: A100
# CU_ARCH=86: RTX 3090
NVCCFLAGS=-std=c++17 $(CU_OPT) $(DEFINES) -Wno-deprecated-gpu-targets -Wno-deprecated-declarations
ifneq ($(CU_ARCH),)
NVCCFLAGS+= --generate-code arch=compute_${CU_ARCH},code=sm_${CU_ARCH} --compiler-options -fPIC  $(DEFINES)
endif
CU_INC=
CU_LIB=-L$(CUDA_DIR)/lib64 -lcudart -lcublas -lcusparse -lcurand 

else

NVCC=$(CC)
NVCCFLAGS=$(CFLAGS)
CU_INC=$(INC)
CU_LIB=$(LIB)
endif

LINKER=mpicxx

