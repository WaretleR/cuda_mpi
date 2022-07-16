all: main clean

main: mainMPI.o mainCUDA.o
	mpicxx mainMPI.o mainCUDA.o -L/usr/local/cuda/lib64 -lcudart -o main

mainCUDA.o: mainCUDA.cu
	nvcc -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -c mainCUDA.cu -o mainCUDA.o

mainMPI.o: mainMPI.c
	mpicxx -lm -c mainMPI.c -o mainMPI.o

clean:
	rm *.o
