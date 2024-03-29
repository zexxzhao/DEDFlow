
###############################################################################
# Prerequisites: HDF5, CUDA (runtime, cublas) 
###############################################################################

# HDF5: https://www.hdfgroup.org/downloads/hdf5/
# CUDA: https://developer.nvidia.com/cuda-downloads

ifeq ($(origin HDF5_ROOT), undefined)
$(error HDF5_ROOT is not set)
endif

MK_HDF5_DIR=$(HDF5_ROOT)

ifeq ($(origin CUDA_ROOT), undefined)
CUDA_DIR=/usr/local/cuda
else
CUDA_DIR=$(CUDA_ROOT)
endif


###############################################################################
# Compiler and flags
###############################################################################
CC=gcc
CFLAGS=-O0 -g3 -std=c99 \
			 -Wall -Wextra -Werror \
			 -Wconversion -Wdouble-promotion \
			 -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion \
			 -fsanitize=address,undefined -fno-omit-frame-pointer
INC=-I$(MK_HDF5_DIR)/include -I$(CUDA_DIR)/include
LIB=-L$(MK_HDF5_DIR)/lib -lhdf5 -lm

NVCC=nvcc
NVCCFLAGS=-O0 -g -G -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75
CU_INC=
CU_LIB=-L$(CUDA_DIR)/lib64 -lcudart -lcublas

DEST=bin
