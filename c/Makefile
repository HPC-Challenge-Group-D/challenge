# Compilers
MPICC= mpic++
NVCC= nvcc

NVCC_FLAGS = -Xcompiler -march=native -arch=sm_80
MPICXX_FLAGS = -DUSE_NVTX
LD_FLAGS = -lcudart -lnvToolsExt

# Commands
all: jacobiSolver

reduction.o: reduction.cu reduction.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

jacobi.o: jacobi.cu jacobi.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

init.o: init.cu init.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

jacobiSolver: main.c jacobi.o init.o reduction.o
	$(MPICC) $(MPICXX_FLAGS) $^ $(LD_FLAGS) -o $@

clean:
	rm -rf *.o *~
