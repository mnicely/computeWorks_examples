EXE			= computeWorks_mm
CUDA		= 10.1
PGI			= 19.10
CUDA_PATH 	= /usr/local/cuda-$(CUDA)
PGI_PATH	= /opt/pgi/linux86-64-nollvm/$(PGI)
CUDAC		= $(CUDA_PATH)/bin/nvcc
CXX			= $(PGI_PATH)/bin/pgc++
OBJ			= o
NVCCFLAGS	= -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
CXXFLAGS	= -O2
ACCLINK		= -Xlinker "-Bstatic -lblas -Bdynamic"
ACCFLAGS	= -Xcompiler "-V$(PGI) -Bstatic_pgi -acc -mp -Mcuda -Minfo=accel -ta=tesla:nordc -ta=time"
LDFLAGS		= -L$(PGI_PATH)/lib
INCLUDE		=
LIBS		= -lcublas

all: computeWorks_mm 

computeWorks_mm: src/computeWorks_mm.cu
	$(CUDAC) -x cu -rdc=true -ccbin $(CXX) $(CXXFLAGS) $(LDFLAGS) $(LIBS) $(ACCLINK) $(ACCFLAGS) $(NVCCFLAGS) -o $(EXE) $<

clean:
	@echo 'Cleaning up...'
	@rm -rf $(EXE) *.$(OBJ)
