
NVCC        = nvcc

NVCC_FLAGS  = --ptxas-options=-v -I/usr/local/cuda/include -gencode=arch=compute_50,code=\"sm_50,compute_50\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = matrixmul
OBJ	        = matrixmul_cu.o matrixmul_cpp.o

default: $(EXE)

matrixmul_cu.o: matrixmul.cu matrixmul_kernel.cu matrixmul.h
	$(NVCC) -c -o $@ matrixmul.cu $(NVCC_FLAGS)

matrixmul_cpp.o: matrixmul_gold.cpp
	$(NVCC) -c -o $@ matrixmul_gold.cpp $(NVCC_FLAGS) 

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
