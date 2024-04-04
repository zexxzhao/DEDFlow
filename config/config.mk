
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
			 -Wall -Wextra \
			 -Wconversion -Wdouble-promotion \
			 -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion \
			 -Wno-deprecated-declarations \
			 -fsanitize=undefined # -fno-omit-frame-pointer
INC=-I$(MK_HDF5_DIR)/include -I$(CUDA_DIR)/include
LIB=-L$(MK_HDF5_DIR)/lib -lhdf5 -lm

NVCC=nvcc
NVCCFLAGS=-O0 -g -G
CU_INC=
CU_LIB=-L$(CUDA_DIR)/lib64 -lcudart -lcublas -lcusparse

LINKER=g++

DEST=bin
