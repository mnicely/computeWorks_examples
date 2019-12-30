NVCC		:=nvcc
CPP			:=g++
PGI_V		:=19.10
PGI			:=/opt/pgi/linux86-64-llvm/$(PGI_V)/bin/pgc++
CFLAGS		:=-O3
NFLAGS		:=-O3
OFLAGS		:=-O3 -fopenmp
AFLAGS		:=-V$(PGI_V) -acc -Mcuda -Minfo=accel -ta=tesla:cc70 -ta=time
ARCHES		:=-gencode arch=compute_70,code=\"compute_70,sm_70\"
INC_DIR		:=
LIB_DIR		:=
SRC_DIR		:= src

SOURCES = 	cuda \
			blas \
			cublas \
			serial \
			openacc \
			openmp

all: $(SOURCES)
.PHONY: all

cuda: $(SRC_DIR)/cuda.cu
	$(NVCC) $(NFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)
	
blas: $(SRC_DIR)/blas.cpp
	$(CPP) $(CFLAGS) $(INC_DIR) $(LIB_DIR)$^ -o $@ -lopenblas
	
cublas: $(SRC_DIR)/cublas.cpp
	$(NVCC) -x cu $(NFLAGS) $(INC_DIR) $(LIB_DIR)$^ -o $@ -lcublas
	
serial: $(SRC_DIR)/serial.cpp
	$(CPP) $(CFLAGS) $(INC_DIR) $(LIB_DIR)$^ -o $@ $(LIBS)
	
openacc: $(SRC_DIR)/openacc.cpp
	$(PGI) $(AFLAGS) $^ -o $@
	
openmp: $(SRC_DIR)/openmp.cpp
	$(CPP) $(OFLAGS) $(INC_DIR) $(LIB_DIR) $^ -o $@ $(LIBS)
	
clean:
	@echo 'Cleaning up...'
	@rm -rf $(SOURCES)
