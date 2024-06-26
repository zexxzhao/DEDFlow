
###############################################################################
# Prerequisites: HDF5, CUDA (runtime, cublas, cusparse), METIS
###############################################################################

# HDF5: https://www.hdfgroup.org/downloads/hdf5/
# CUDA: https://developer.nvidia.com/cuda-downloads
# METIS: http://glaros.dtc.umn.edu/gkhome/metis/metis/download

ifeq ($(origin HDF5_ROOT), undefined)
$(error HDF5_ROOT is not set)
endif

MK_HDF5_DIR=$(HDF5_ROOT)

ifeq ($(origin CUDA_ROOT), undefined)
CUDA_DIR=/usr/local/cuda
else
CUDA_DIR=$(CUDA_ROOT)
endif

ifeq ($(origin METIS_DIR), undefined)
$(warning METIS_DIR is not set)
$(warning METIS will not be used)
MK_METIS_DIR=
else
MK_METIS_DIR=$(METIS_DIR)
endif


###############################################################################
# Compiler and flags
###############################################################################
FLAGS=-O3 \
			-Wall -Wextra \
			-Wconversion -Wdouble-promotion \
			-Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion \
			-Wno-deprecated-declarations
DEFINES=-DUSE_I32_INDEX -DUSE_F64_VALUE
CC=gcc
CFLAGS=-std=c99 -g3 $(FLAGS) $(DEFINES) -fsanitize=undefined # -fno-omit-frame-pointer

INC=-I$(CUDA_DIR)/include
LIB=-lm

INC+=-I$(MK_HDF5_DIR)/include
LIB+=-L$(MK_HDF5_DIR)/lib -lhdf5

# if MK_METIS_DIR is not set, then METIS will not be used
ifneq ($(MK_METIS_DIR),)
CLFAGS+= -DUSE_METIS
INC+=-I$(MK_METIS_DIR)/include
LIB+=-L$(MK_METIS_DIR)/lib -lmetis
endif

NVCC=nvcc
NVCCFLAGS=-O3 -g -G -std=c++17 $(DEFINES) -Wno-deprecated-gpu-targets -Wno-deprecated-declarations
NVCCFLAGS+= --generate-code arch=compute_75,code=sm_75 # RTX 2080 Ti
NVCCFLAGS+= --generate-code arch=compute_86,code=sm_86 # RTX 3090
CU_INC=
CU_LIB=-L$(CUDA_DIR)/lib64 -lcudart -lcublas -lcusparse -lcurand

LINKER=g++

DEST=bin
