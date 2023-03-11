# Compilers
CC= mpic++

CFLAGS = -lcudart -DUSE_NVTX -lcudadevrt -gpu=cc80 -O2

# Commands
all: jacobiSolver

jacobiSolver: main.cu solver.cu
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -rf *.o *~
