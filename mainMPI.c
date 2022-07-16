#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#ifndef MAX_STEPS
#define MAX_STEPS 20
#endif
#ifndef T
#define T 1.0
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void runFirst(double **selfBorders,
    int xLen, int yLen, int zLen, 
    int xStart, int yStart, int zStart, 
    double hx, double hy, double hz, 
    double scaleCoef,
    double Lx, double Ly, double Lz,
    int K);

void runSecond(double **selfBorders, double **borders,
    int xLen, int yLen, int zLen,
    int xStart, int yStart, int zStart,
    int xBlocks, int yBlocks, int zBlocks,
    int i, int j, int k,
    double hx, double hy, double hz, 
    double scaleCoef, int stepNumber,
    double Lx, double Ly, double Lz,
    int K);

void runMain(double **selfBorders, double **borders,
    int xLen, int yLen, int zLen,
    int xStart, int yStart, int zStart,
    int xBlocks, int yBlocks, int zBlocks,
    int i, int j, int k,
    double hx, double hy, double hz, 
    double scaleCoef, int stepNumber,
    double Lx, double Ly, double Lz,
    int K);

void runBorders(double **borders,
    int xLen, int yLen, int zLen,
    int xBlocks, int yBlocks, int zBlocks,
    int i, int j, int k,
    int stepNumber);

double runErrors(double *out,
    int xLen, int yLen, int zLen,
    int xStart, int yStart, int zStart,
    int i, int j, int k,
    double hx, double hy, double hz, 
    double scaleCoef, int stepNumber,
    double Lx, double Ly, double Lz,
    int K);

double Lx = 1, Ly = 1, Lz = 1;
int N = 128;
int K = 256;

double analytical(double x, double y, double z, double t)
{
    return sin(2 * M_PI / Lx * x) * sin(M_PI / Ly * y + M_PI) * sin(2 * M_PI / Lz * z + 2 * M_PI) * 
        cos(M_PI * sqrt(4 / (Lx * Lx) + 1 / (Ly * Ly) + 4 / (Lz * Lz)) * t + M_PI);
}

double getItem(double *matrix, double **borders, int xLen, int yLen, int zLen, int i, int j, int k)
{
    if (i < 0)
    {
        return borders[0][j * zLen + k];
    }
    if (i >= xLen)
    {
        return borders[1][j * zLen + k];
    }
    if (j < 0)
    {
        return borders[2][i * zLen + k];
    }
    if (j >= yLen)
    {
        return borders[3][i * zLen + k];
    }
    if (k < 0)
    {
        return borders[4][i * yLen + j];
    }
    if (k >= zLen)
    {
        return borders[5][i * yLen + j];
    }
    return matrix[i * yLen * zLen + j * zLen + k];
}

int main(int argc, char **argv)
{
    if (argc > 1)
    {
        N = atoi(argv[1]);
    }
    if (argc > 2)
    {
        K = atoi(argv[2]);
    }
    if (argc > 5)
    {
        Lx = atof(argv[3]);
        Ly = atof(argv[4]);
        Lz = atof(argv[5]);
    }

    int size, rank;
    int i, j, k;
    int xBlocks, yBlocks, zBlocks;
    double hx = Lx / N, hy = Ly / N, hz = Lz / N;
    double scaleCoef = N * 1.0 / (N - 1);
    double startTime, workTime, totalTime;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    startTime = MPI_Wtime();

    xBlocks = (int)floor(pow(size, 1.0 / 3.0));
    while (size % xBlocks > 0)
    {
        xBlocks = xBlocks - 1;
    }
    yBlocks = (int)floor(sqrt(size / xBlocks));
    while ((size / xBlocks) % yBlocks > 0)
    {
        yBlocks = yBlocks - 1;
    }
    zBlocks = size / xBlocks / yBlocks;

    i = rank / (yBlocks * zBlocks);
    j = (rank - i * yBlocks * zBlocks) / zBlocks;
    k = rank - i * yBlocks * zBlocks - j * zBlocks;

    //индексы старта блоков и длины блоков в узлах
    int xStart = i * (int)floor(N / xBlocks);
    int xLen;
    if (i >= N % xBlocks)
    {
        xStart += N % xBlocks;
        xLen = (int)floor(N / xBlocks);
    }
    else
    {
        xStart += i;
        xLen = (int)floor(N / xBlocks) + 1;
    }

    int yStart = j * (int)floor(N / yBlocks);
    int yLen;
    if (j >= N % xBlocks)
    {
        yStart += N % yBlocks;
        yLen = (int)floor(N / yBlocks);
    }
    else
    {
        yStart += j;
        yLen = (int)floor(N / yBlocks) + 1;
    }

    int zStart = k * (int)floor(N / zBlocks);
    int zLen;
    if (k >= N % zBlocks)
    {
        zStart += N % zBlocks;
        zLen = (int)floor(N / zBlocks);
    }
    else
    {
        zStart += k;
        zLen = (int)floor(N / zBlocks) + 1;
    }

    double error = 0;
    double maxError = 0;

    double *curr;
    curr = (double *)malloc(sizeof(double) * xLen * yLen * zLen);

    double **borders = (double **)malloc(sizeof(double *) * 6); //x0, xN, y0, yN, z0, zN
    double **selfBorders = (double **)malloc(sizeof(double *) * 6); //x0, xN, y0, yN, z0, zN
    for (int l = 0; l < 2; ++l)
    {
        borders[l] = (double *)malloc(sizeof(double) * yLen * zLen);
        borders[l+2] = (double *)malloc(sizeof(double) * xLen * zLen);
        borders[l+4] = (double *)malloc(sizeof(double) * xLen * yLen);
        selfBorders[l] = (double *)malloc(sizeof(double) * yLen * zLen);
        selfBorders[l+2] = (double *)malloc(sizeof(double) * xLen * zLen);
        selfBorders[l+4] = (double *)malloc(sizeof(double) * xLen * yLen);
    }

    for (int stepNumber = 0; stepNumber < MAX_STEPS; ++stepNumber)
    {
        if (stepNumber == 0)
        {
            runFirst(selfBorders,
                xLen, yLen, zLen,
                xStart, yStart, zStart,
                hx, hy, hz,
                scaleCoef,
                Lx, Ly, Lz,
                K);
        }
        else if (stepNumber == 1)
        {
            runSecond(selfBorders, borders,
                xLen, yLen, zLen,
                xStart, yStart, zStart,
                xBlocks, yBlocks, zBlocks,
                i, j, k,
                hx, hy, hz, 
                scaleCoef, stepNumber,
                Lx, Ly, Lz,
                K);
        }
        else
        {
            runMain(selfBorders, borders,
                xLen, yLen, zLen,
                xStart, yStart, zStart,
                xBlocks, yBlocks, zBlocks,
                i, j, k,
                hx, hy, hz, 
                scaleCoef, stepNumber,
                Lx, Ly, Lz,
                K);
        }


        //обмен границами

        MPI_Request xPrevReq, xNextReq, yPrevReq, yNextReq, zPrevReq, zNextReq;

        if (i == 0)
        {
            MPI_Isend(selfBorders[0], yLen * zLen, MPI_DOUBLE, rank - yBlocks * zBlocks + size, 1, MPI_COMM_WORLD, &xPrevReq);
        }
        else
        {
            MPI_Isend(selfBorders[0], yLen * zLen, MPI_DOUBLE, rank - yBlocks * zBlocks, 1, MPI_COMM_WORLD, &xPrevReq);
        }

        if (i == xBlocks - 1)
        {            
            MPI_Isend(selfBorders[1], yLen * zLen, MPI_DOUBLE, rank + yBlocks * zBlocks - size, 0, MPI_COMM_WORLD, &xNextReq);
        }
        else
        {
            MPI_Isend(selfBorders[1], yLen * zLen, MPI_DOUBLE, rank + yBlocks * zBlocks, 0, MPI_COMM_WORLD, &xNextReq);
        }
    
        if (j > 0)
        {
            MPI_Isend(selfBorders[2], xLen * zLen, MPI_DOUBLE, rank - zBlocks, 3, MPI_COMM_WORLD, &yPrevReq);
        }

        if (j < yBlocks - 1)
        {
            MPI_Isend(selfBorders[3], xLen * zLen, MPI_DOUBLE, rank + zBlocks, 2, MPI_COMM_WORLD, &yNextReq);
        }

        if (k == 0)
        {
            MPI_Isend(selfBorders[4], xLen * yLen, MPI_DOUBLE, rank - 1 + zBlocks, 5, MPI_COMM_WORLD, &zPrevReq);
        }
        else
        {
            MPI_Isend(selfBorders[4], xLen * yLen, MPI_DOUBLE, rank - 1, 5, MPI_COMM_WORLD, &zPrevReq);
        }

        if (k == zBlocks - 1)
        {
            MPI_Isend(selfBorders[5], xLen * yLen, MPI_DOUBLE, rank + 1 - zBlocks, 4, MPI_COMM_WORLD, &zNextReq);
        }
        else
        {
            MPI_Isend(selfBorders[5], xLen * yLen, MPI_DOUBLE, rank + 1, 4, MPI_COMM_WORLD, &zNextReq);
        }


        //получение границ от соседних блоков

        MPI_Status xPrevStatus, xNextStatus, yPrevStatus, yNextStatus, zPrevStatus, zNextStatus;

        if (i == 0)
        {
            MPI_Recv(borders[0], yLen * zLen, MPI_DOUBLE, rank - yBlocks * zBlocks + size, 0, MPI_COMM_WORLD, &xPrevStatus);
        }
        else
        {
            MPI_Recv(borders[0], yLen * zLen, MPI_DOUBLE, rank - yBlocks * zBlocks, 0, MPI_COMM_WORLD, &xPrevStatus);
        }

        if (i == xBlocks - 1)
        {
            MPI_Recv(borders[1], yLen * zLen, MPI_DOUBLE, rank + yBlocks * zBlocks - size, 1, MPI_COMM_WORLD, &xNextStatus);
        }
        else
        {
            MPI_Recv(borders[1], yLen * zLen, MPI_DOUBLE, rank + yBlocks * zBlocks, 1, MPI_COMM_WORLD, &xNextStatus);
        }

        if (j > 0)
        {
            MPI_Recv(borders[2], xLen * zLen, MPI_DOUBLE, (rank - zBlocks + size) % size, 2, MPI_COMM_WORLD, &yPrevStatus);
        }
        
        if (j < yBlocks - 1)
        {
            MPI_Recv(borders[3], xLen * zLen, MPI_DOUBLE, (rank + zBlocks + size) % size, 3, MPI_COMM_WORLD, &yNextStatus);
        }

        if (k == 0)
        {
            MPI_Recv(borders[4], xLen * yLen, MPI_DOUBLE, rank - 1 + zBlocks, 4, MPI_COMM_WORLD, &zPrevStatus);
        }
        else
        {
            MPI_Recv(borders[4], xLen * yLen, MPI_DOUBLE, rank - 1, 4, MPI_COMM_WORLD, &zPrevStatus);
        }
    
        if (k == zBlocks - 1)
        {
            MPI_Recv(borders[5], xLen * yLen, MPI_DOUBLE, rank + 1 - zBlocks, 5, MPI_COMM_WORLD, &zNextStatus);
        }
        else
        {
            MPI_Recv(borders[5], xLen * yLen, MPI_DOUBLE, rank + 1, 5, MPI_COMM_WORLD, &zNextStatus);
        }

        MPI_Barrier(MPI_COMM_WORLD);    
  

        //вычисление границ сетки

        runBorders(borders,
            xLen, yLen, zLen,
            xBlocks, yBlocks, zBlocks,
            i, j, k,
            stepNumber);
    }

    //получение максимальной ошибки в блоке, получение итоговой матрицы и освобождение памяти GPU
    error = runErrors(curr,
        xLen, yLen, zLen,
        xStart, yStart, zStart,
        i, j, k,
        hx, hy, hz, 
        scaleCoef, MAX_STEPS - 1,
        Lx, Ly, Lz,
        K);
    
    //сбор значений ошибок
    MPI_Reduce(&error, &maxError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("Error on step %d = %f\n", MAX_STEPS - 1, maxError);
    }

    workTime = MPI_Wtime() - startTime;
    MPI_Reduce(&workTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("Processes = %d; PointsPerDimension = %d; Lx = %f; xBlocks = %d; yBlocks = %d; zBlocks = %d; Time = %f\n", size, N, Lx,  xBlocks, yBlocks, zBlocks, totalTime);
    }

    //вывод значений ошибок во всех узлах на последнем шаге

    char *fileName = malloc(sizeof(char) * 10);
    snprintf(fileName, 10, "%d.txt", rank);
    FILE *fptr = fopen(fileName, "w");
    for (int l = 0; l < xLen; ++l)
    {
        fprintf(fptr, "x %d\n", xStart + l);
        for (int m = 0; m < yLen; ++m)
        {
            fprintf(fptr, "y %d\n", yStart + m);
            fprintf(fptr, "z %d\n", zStart);
            for (int n = 0; n < zLen; ++n)
            {
                fprintf(fptr, "%f : %f ;", curr[l * yLen * zLen + m * zLen + n], analytical(scaleCoef * (xStart + l) * hx, scaleCoef * (yStart + m) * hy, scaleCoef * (zStart + n) * hz, (MAX_STEPS - 1) * T / K));
            }
            fprintf(fptr, "\n");
        }
    }
    fclose(fptr);
    free(fileName);

    free(curr);

    for (int l = 0; l < 6; ++l)
    {
        free(borders[l]);
        free(selfBorders[l]);
    }

    free(borders);
    free(selfBorders);

    MPI_Finalize();
}
