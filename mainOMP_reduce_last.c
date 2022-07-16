#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#define MAX_STEPS 20
#define T 1.0
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double Lx = 1, Ly = 1, Lz = 1;
int N = 128;
int K = 256;

double analytical(double x, double y, double z, double t)
{
    return sin(2 * M_PI / Lx * x) * sin(M_PI / Ly * y + M_PI) * sin(2 * M_PI / Lz * z + 2 * M_PI) * 
        cos(M_PI * sqrt(4 / (Lx * Lx) + 1 / (Ly * Ly) + 4 / (Lz * Lz)) * t + M_PI);
}

double getItem(double ***matrix, double **borders, int xLen, int yLen, int zLen, int i, int j, int k)
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
    return matrix[i][j][k];
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

    double ***prev, ***curr;
    prev = (double ***)malloc(sizeof(double **) * xLen);
    curr = (double ***)malloc(sizeof(double **) * xLen);
    
    #pragma omp parallel for
    for (int l = 0; l < xLen; ++l)
    {
        prev[l] = (double **)malloc(sizeof(double *) * yLen);
        curr[l] = (double **)malloc(sizeof(double *) * yLen);
        for (int m = 0; m < yLen; ++m)
        {
            prev[l][m] = (double *)malloc(sizeof(double) * zLen);
            curr[l][m] = (double *)malloc(sizeof(double) * zLen);
        }
    }

    double **borders = (double **)malloc(sizeof(double *) * 6); //x0, xN, y0, yN, z0, zN
    for (int l = 0; l < 2; ++l)
    {
        borders[l] = (double *)malloc(sizeof(double) * yLen * zLen);
        borders[l+2] = (double *)malloc(sizeof(double) * xLen * zLen);
        borders[l+4] = (double *)malloc(sizeof(double) * xLen * yLen);
    }

    double error = 0;
    double maxError = 0;

    for (int stepNumber = 0; stepNumber < MAX_STEPS; ++stepNumber)
    {
        if (stepNumber == 0)
        {
            #pragma omp parallel for
            for (int l = 0; l < xLen; ++l)
            {
                for (int m = 0; m < yLen; ++m)
                {
                    for (int n = 0; n < zLen; ++n)
                    {
                        prev[l][m][n] = analytical(scaleCoef * (xStart + l) * hx, scaleCoef * (yStart + m) * hy, scaleCoef * (zStart + n) * hz, 0);
                    }
                }
            }
        }
        else if (stepNumber == 1)
        {
            #pragma omp parallel for
            for (int l = 0; l < xLen; ++l)
            {
                for (int m = 0; m < yLen; ++m)
                {
                    for (int n = 0; n < zLen; ++n)
                    {
                        if (xStart + l == 0 || i == xBlocks - 1 && l == xLen - 1 || 
                            yStart + m == 0 || j == yBlocks - 1 && m == yLen - 1 || 
                            zStart + n == 0 || k == zBlocks - 1 && n == zLen - 1)
                        {
                            prev[l][m][n] = analytical(scaleCoef * (xStart + l) * hx, scaleCoef * (yStart + m) * hy, scaleCoef * (zStart + n) * hz, stepNumber * T / K);
                        }
                        else
                        {
                            prev[l][m][n] = curr[l][m][n] + 
                                (T / K) * (T / K) / 2 * (
                                    (getItem(curr, borders, xLen, yLen, zLen, l - 1, m, n) - 2 * getItem(curr, borders, xLen, yLen, zLen, l, m, n) + getItem(curr, borders, xLen, yLen, zLen, l + 1, m, n)) / (hx * hx) + 
                                    (getItem(curr, borders, xLen, yLen, zLen, l, m - 1, n) - 2 * getItem(curr, borders, xLen, yLen, zLen, l, m, n) + getItem(curr, borders, xLen, yLen, zLen, l, m + 1, n)) / (hy * hy) + 
                                    (getItem(curr, borders, xLen, yLen, zLen, l, m, n - 1) - 2 * getItem(curr, borders, xLen, yLen, zLen, l, m, n) + getItem(curr, borders, xLen, yLen, zLen, l, m, n + 1)) / (hz * hz)
                                );
                        }
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for (int l = 0; l < xLen; ++l)
            {
                for (int m = 0; m < yLen; ++m)
                {
                    for (int n = 0; n < zLen; ++n)
                    {
                        if (xStart + l != 0 && (i != xBlocks - 1 || l != xLen - 1) && 
                            yStart + m != 0 && (j != yBlocks - 1 || m != yLen - 1) &&
                            zStart + n != 0 && (k != zBlocks - 1 || n != zLen - 1))
                        {
                            prev[l][m][n] = 2 * curr[l][m][n] - prev[l][m][n] + 
                                (T / K) * (T / K) * (
                                    (getItem(curr, borders, xLen, yLen, zLen, l - 1, m, n) - 2 * getItem(curr, borders, xLen, yLen, zLen, l, m, n) + getItem(curr, borders, xLen, yLen, zLen, l + 1, m, n)) / (hx * hx) + 
                                    (getItem(curr, borders, xLen, yLen, zLen, l, m - 1, n) - 2 * getItem(curr, borders, xLen, yLen, zLen, l, m, n) + getItem(curr, borders, xLen, yLen, zLen, l, m + 1, n)) / (hy * hy) + 
                                    (getItem(curr, borders, xLen, yLen, zLen, l, m, n - 1) - 2 * getItem(curr, borders, xLen, yLen, zLen, l, m, n) + getItem(curr, borders, xLen, yLen, zLen, l, m, n + 1)) / (hz * hz)
                                );
                        }
                    }
                }
            }
        }

        double ***buf = prev;
        prev = curr;
        curr = buf;


        //обмен границами

        double *xPrev, *xNext, *yPrev, *yNext, *zPrev, *zNext;
        MPI_Request xPrevReq, xNextReq, yPrevReq, yNextReq, zPrevReq, zNextReq;

        xPrev = (double *)malloc(sizeof(double) * yLen * zLen);
        #pragma omp parallel for
        for (int m = 0; m < yLen; ++m)
        {
            for (int n = 0; n < zLen; ++n)
            {
                if (i == 0)
                {
                    xPrev[m * zLen + n] = curr[1][m][n];
                }
                else 
                {
                    xPrev[m * zLen + n] = curr[0][m][n];
                }
            }
        }
        if (i == 0)
        {
            MPI_Isend(xPrev, yLen * zLen, MPI_DOUBLE, rank - yBlocks * zBlocks + size, 1, MPI_COMM_WORLD, &xPrevReq);
        }
        else
        {
            MPI_Isend(xPrev, yLen * zLen, MPI_DOUBLE, rank - yBlocks * zBlocks, 1, MPI_COMM_WORLD, &xPrevReq);
        }

        xNext = (double *)malloc(sizeof(double) * yLen * zLen);
        #pragma omp parallel for
        for (int m = 0; m < yLen; ++m)
        {
            for (int n = 0; n < zLen; ++n)
            {
                if (i == xBlocks - 1)
                {
                    xNext[m * zLen + n] = curr[xLen - 2][m][n];
                }
                else
                {
                    xNext[m * zLen + n] = curr[xLen - 1][m][n];
                }
            }
        }
        if (i == xBlocks - 1)
        {            
            MPI_Isend(xNext, yLen * zLen, MPI_DOUBLE, rank + yBlocks * zBlocks - size, 0, MPI_COMM_WORLD, &xNextReq);
        }
        else
        {
            MPI_Isend(xNext, yLen * zLen, MPI_DOUBLE, rank + yBlocks * zBlocks, 0, MPI_COMM_WORLD, &xNextReq);
        }
    
        if (j > 0)
        {
            yPrev = (double *)malloc(sizeof(double) * xLen * zLen);
            #pragma omp parallel for
            for (int l = 0; l < xLen; ++l)
            {
                for (int n = 0; n < zLen; ++n)
                {
                    yPrev[l * zLen + n] = curr[l][0][n];
                }
            }
            MPI_Isend(yPrev, xLen * zLen, MPI_DOUBLE, rank - zBlocks, 3, MPI_COMM_WORLD, &yPrevReq);
        }

        if (j < yBlocks - 1)
        {
            yNext = (double *)malloc(sizeof(double) * xLen * zLen);
            #pragma omp parallel for
            for (int l = 0; l < xLen; ++l)
            {
                for (int n = 0; n < zLen; ++n)
                {
                    yNext[l * zLen + n] = curr[l][yLen - 1][n];
                }
            }
            MPI_Isend(yNext, xLen * zLen, MPI_DOUBLE, rank + zBlocks, 2, MPI_COMM_WORLD, &yNextReq);
        }


        zPrev = (double *)malloc(sizeof(double) * xLen * yLen);
        #pragma omp parallel for
        for (int l = 0; l < xLen; ++l)
        {
            for (int m = 0; m < yLen; ++m)
            {
                if (k == 0)
                {
                    zPrev[l * yLen + m] = curr[l][m][1];
                }
                else
                {
                    zPrev[l * yLen + m] = curr[l][m][0];
                }
            }
        }
        if (k == 0)
        {
            MPI_Isend(zPrev, xLen * yLen, MPI_DOUBLE, rank - 1 + zBlocks, 5, MPI_COMM_WORLD, &zPrevReq);
        }
        else
        {
            MPI_Isend(zPrev, xLen * yLen, MPI_DOUBLE, rank - 1, 5, MPI_COMM_WORLD, &zPrevReq);
        }

        zNext = (double *)malloc(sizeof(double) * xLen * yLen);
        #pragma omp parallel for
        for (int l = 0; l < xLen; ++l)
        {
            for (int m = 0; m < yLen; ++m)
            {
                if (k == zBlocks - 1)
                {
                    zNext[l * yLen + m] = curr[l][m][zLen - 2];
                }
                else
                {
                    zNext[l * yLen + m] = curr[l][m][zLen - 1];
                }
            }
        }
        if (k == zBlocks - 1)
        {
            MPI_Isend(zNext, xLen * yLen, MPI_DOUBLE, rank + 1 - zBlocks, 4, MPI_COMM_WORLD, &zNextReq);
        }
        else
        {
            MPI_Isend(zNext, xLen * yLen, MPI_DOUBLE, rank + 1, 4, MPI_COMM_WORLD, &zNextReq);
        }


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

        //освобождение буферов для пересылки границ после блокировки

        free(xPrev);       
        free(xNext);
        if (j > 0)
        {
            free(yPrev);
        }
        if (j < yBlocks - 1)
        {
            free(yNext);
        }
        free(zPrev);
        free(zNext);    
        
        //вычисление границ

        if (stepNumber >= 2)
        {
            if (i == 0)
            {
                #pragma omp parallel for
                for (int m = 0; m < yLen; ++m)
                {
                    for (int n = 0; n < zLen; ++n)
                    {
                        curr[0][m][n] = (curr[1][m][n] + borders[0][m * zLen + n]) / 2;
                    }
                }
            }

            if (i == xBlocks - 1)
            {
                #pragma omp parallel for
                for (int m = 0; m < yLen; ++m)
                {
                    for (int n = 0; n < zLen; ++n)
                    {
                        curr[xLen - 1][m][n] = (curr[xLen - 2][m][n] + borders[1][m * zLen + n]) / 2;
                    }
                }
            }

            if (j == 0)
            {
                #pragma omp parallel for
                for (int l = 0; l < xLen; ++l)
                {
                    for (int n = 0; n < zLen; ++n)
                    {
                        curr[l][0][n] = 0;
                    }
                }
            }
            
            if (j == yBlocks - 1)
            {
                #pragma omp parallel for
                for (int l = 0; l < xLen; ++l)
                {
                    for (int n = 0; n < zLen; ++n)
                    {
                        curr[l][yLen - 1][n] = 0;
                    }
                }
            }

            if (k == 0)
            {
                #pragma omp parallel for
                for (int l = 0; l < xLen; ++l)
                {
                    for (int m = 0; m < yLen; ++m)
                    {
                        curr[l][m][0] = (curr[l][m][1] + borders[4][l * yLen + m]) / 2;
                    }
                }
            }
            
            if (k == zBlocks - 1)
            {
                #pragma omp parallel for
                for (int l = 0; l < xLen; ++l)
                {
                    for (int m = 0; m < yLen; ++m)
                    {
                        curr[l][m][zLen - 1] = (curr[l][m][zLen - 2] + borders[5][l * yLen + m]) / 2;
                    }
                }
            }
        }
    }
    #pragma omp parallel for
    for (int l = 0; l < xLen; ++l)
    {
        for (int m = 0; m < yLen; ++m)
        {
            for (int n = 0; n < zLen; ++n)
            {
                double localError = fabs(curr[l][m][n] - analytical(scaleCoef * (xStart + l) * hx, scaleCoef * (yStart + m) * hy, scaleCoef * (zStart + n) * hz, (MAX_STEPS - 1) * T / K));
		#pragma omp critical
                if (localError > error)
                {
                    error = localError;
                }
            }
        }
    }

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
                fprintf(fptr, "%f : %f ;", curr[l][m][n], analytical(scaleCoef * (xStart + l) * hx, scaleCoef * (yStart + m) * hy, scaleCoef * (zStart + n) * hz, (MAX_STEPS - 1) * T / K));
            }
            fprintf(fptr, "\n");
        }
    }
    fclose(fptr);
    free(fileName);

    MPI_Finalize();
}
