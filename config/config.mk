
###############################################################################
# Prerequisites: HDF5, CUDA (runtime, cublas, cusparse), METIS
###############################################################################

# HDF5: https://www.hdfgroup.org/downloads/hdf5/
# HYPRE: https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/download
# CUDA: https://developer.nvidia.com/cuda-downloads
# METIS: http://glaros.dtc.umn.edu/gkhome/metis/metis/download

ifeq ($(origin HDF5_ROOT), undefined)
$(error HDF5_ROOT is not set)
endif

ifeq ($(origin HYPRE_DIR), undefined)
$(error HYPRE_DIR is not set)
endif

ifeq ($(origin CUDA_ROOT), undefined)
CUDA_DIR=/usr/local/cuda
else
CUDA_DIR=$(CUDA_ROOT)
endif

ifeq ($(origin AMGX_DIR), undefined)
$(warning AMGX_DIR is not set)
endif

ifeq ($(origin METIS_DIR), undefined)
$(warning METIS_DIR is not set)
$(warning METIS will not be used)
MK_METIS_DIR=
else
MK_METIS_DIR=$(METIS_DIR)
endif

ifndef PREFIX
PREFIX=$(PWD)
endif

###############################################################################
# Compiler and flags
###############################################################################
DEBUG=1
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
DEFINES=-DUSE_I32_INDEX -DUSE_F64_VALUE # -DDBG_TET
CC=mpicc
CFLAGS=-std=c99 $(FLAGS) $(DEFINES) $(C_OPT) # -fno-omit-frame-pointer

INC=-I$(CUDA_DIR)/include
LIB=-lm

INC+=-I$(HDF5_ROOT)/include -I$(HYPRE_DIR)/include
LIB+=-L$(HDF5_ROOT)/lib -lhdf5 -L$(HYPRE_DIR)/lib -lHYPRE

# if MK_METIS_DIR is not set, then METIS will not be used
ifneq ($(MK_METIS_DIR),)
DEFINES+= -DUSE_METIS
INC+=-I$(MK_METIS_DIR)/include
LIB+=-L$(MK_METIS_DIR)/lib -lmetis
endif

ifneq ($(AMGX_DIR),)
DEFINES+= -DUSE_AMGX
INC+=-I$(AMGX_DIR)/include
LIB+=-L$(AMGX_DIR)/lib -lamgx -lcusolver -lnvToolsExt
endif

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

LINKER=mpicxx

