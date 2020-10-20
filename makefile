CUDA_V		:=10.1
PGI_V		:=19.10
CPP		:=g++
NVCC		:=/usr/local/cuda-$(CUDA_V)/bin/nvcc
PGI		:=/opt/pgi/linux86-64-llvm/$(PGI_V)/bin/pgc++
CFLAGS		:=-O3 -std=c++14
PGIFLAGS	:=-V$(PGI_V) -acc -Mcuda:cuda$(CUDA_V) -Minfo=accel -ta=tesla:cc70 -ta=time
ARCHES		:=-gencode arch=compute_70,code=\"compute_70,sm_70\"
LIBS		:=-lopenblas -lcudart -lcublas
SRCDIR		:=./src
OBJDIR		:=obj

SOURCES := 	computeWorks_mm \
		blas \
		cublas \
		cuda \
		openacc \
		openmp

OBJECTS=$(addprefix $(OBJDIR)/, $(SOURCES:%=%.o))

all: build computeWorks_mm
.PHONY: all

build:	
	@mkdir -p $(OBJDIR)
	
computeWorks_mm: $(OBJECTS)
	$(PGI) $(PGIFLAGS) $(LIBS) $(OBJECTS)  -o $@
	
$(OBJDIR)/computeWorks_mm.o: $(SRCDIR)/computeWorks_mm.cpp
	$(NVCC) -x cu -ccbin $(CPP) $(CFLAGS) -c $^ -o $@
	
$(OBJDIR)/blas.o: $(SRCDIR)/blas.cpp
	$(NVCC) -x cu -ccbin $(CPP) $(CFLAGS) -c $^ -o $@
	
$(OBJDIR)/cublas.o: $(SRCDIR)/cublas.cpp
	$(NVCC) -x cu -ccbin $(CPP) -cudart=static $(CFLAGS) ${ARCHES} -c $^ -o $@ -lcublas
	
$(OBJDIR)/cuda.o: $(SRCDIR)/cuda.cu
	$(NVCC) -ccbin $(CPP) -cudart=static $(CFLAGS) ${ARCHES} -c $^ -o $@
	
$(OBJDIR)/openacc.o: $(SRCDIR)/openacc.cpp
	$(PGI) $(PGIFLAGS) $(CFLAGS) -c $^ -o $@
	
$(OBJDIR)/openmp.o: $(SRCDIR)/openmp.cpp
	$(NVCC) -x cu -ccbin $(CPP) -Xcompiler -fopenmp -c $^ -o $@
	
clean:
	@echo 'Cleaning up...'
	@echo 'rm -rf $(SOURCES) $(OBJDIR)/*.o'
	@rm -rf $(SOURCES) $(OBJDIR)/*.o
