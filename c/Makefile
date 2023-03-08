# Compilers
MPICC= mpicc
MPILD= mpic++
NVCC= nvcc

CFLAGS=-std=c99 -O3 -march=native -Wall

# Commands
all: jacobiSolver

reduction.o: reduction.cu
	$(NVCC) -c $^ -o $@

jacobi.o: jacobi.cu
	$(NVCC) -c $^ -o $@

init.o: init.cu
	$(NVCC) -c $^ -o $@

solver.o: solver.c
	$(MPICC) $(CFLAGS) -c $^ -o $@

main.o: main.c
	$(MPICC) $(CFLAGS) -c $^ -o $@

jacobiSolver: main.o solver.o jacobi.o init.o reduction.o
	$(MPICC) -o $@ $^ -lcudart

clean:
	rm -rf *.o *~
