#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#ifndef MAX_STEPS
#define MAX_STEPS 20
#endif
#ifndef T
#define T 1.0
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double *dOut, *dIn;
double *dxSelfPrev, *dxSelfNext, *dySelfPrev, *dySelfNext, *dzSelfPrev, *dzSelfNext;
double *dxPrev, *dxNext, *dyPrev, *dyNext, *dzPrev, *dzNext;
double *errors;

int matrixLength, xBorderLength, yBorderLength, zBorderLength;

__device__ double getItemGPU(double *matrix, 
    double *xPrev, double *xNext, double *yPrev, double *yNext, double *zPrev, double *zNext,
    int xLen, int yLen, int zLen, 
    int i, int j, int k)
{
    if (i < 0)
    {
        return xPrev[j * zLen + k];
    }
    if (i >= xLen)
    {
        return xNext[j * zLen + k];
    }
    if (j < 0)
    {
        return yPrev[i * zLen + k];
    }
    if (j >= yLen)
    {
        return yNext[i * zLen + k];
    }
    if (k < 0)
    {
        return zPrev[i * yLen + j];
    }
    if (k >= zLen)
    {
        return zNext[i * yLen + j];
    }
    return matrix[i * yLen * zLen + j * zLen + k];
}

__global__ void getFirstIter(double *out, 
    double *xSelfPrev, double *xSelfNext, double *ySelfPrev, double *ySelfNext, double *zSelfPrev, double *zSelfNext,
    int xLen, int yLen, int zLen, 
    int xStart, int yStart, int zStart, 
    double hx, double hy, double hz, 
    double scaleCoef,
    double Lx, double Ly, double Lz,
    int K)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;

    if (l < xLen && m < yLen && n < zLen)
    {
        out[l * yLen * zLen + m * zLen + n] = sin(2 * M_PI / Lx * scaleCoef * (xStart + l) * hx) * 
            sin(M_PI / Ly * scaleCoef * (yStart + m) * hy + M_PI) * 
            sin(2 * M_PI / Lz * scaleCoef * (zStart + n) * hz + 2 * M_PI) * 
            cos(M_PI);

        if (l == 0)
        {
            xSelfPrev[m * zLen + n] = out[l * yLen * zLen + m * zLen + n];
        }
        if (l == xLen - 1)
        {
            xSelfNext[m * zLen + n] = out[l * yLen * zLen + m * zLen + n];
        }
        if (m == 0)
        {
            ySelfPrev[l * zLen + n] = out[l * yLen * zLen + m * zLen + n];
        }
        if (m == yLen - 1)
        {
            ySelfNext[l * zLen + n] = out[l * yLen * zLen + m * zLen + n];
        }
        if (n == 0)
        {
            zSelfPrev[l * yLen + m] = out[l * yLen * zLen + m * zLen + n];
        }
        if (n == zLen - 1)
        {
            zSelfNext[l * yLen + m] = out[l * yLen * zLen + m * zLen + n];
        }
    }    
}

__global__ void getSecondIter(double *out, double *in, 
    double *xSelfPrev, double *xSelfNext, double *ySelfPrev, double *ySelfNext, double *zSelfPrev, double *zSelfNext,
    double *xPrev, double *xNext, double *yPrev, double *yNext, double *zPrev, double *zNext,
    int xLen, int yLen, int zLen,
    int xStart, int yStart, int zStart,
    int xBlocks, int yBlocks, int zBlocks,
    int i, int j, int k,
    double hx, double hy, double hz, 
    double scaleCoef, int stepNumber,
    double Lx, double Ly, double Lz,
    int K)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;

    if (l < xLen && m < yLen && n < zLen)
    {
        if (xStart + l == 0 || i == xBlocks - 1 && l == xLen - 1 || 
            yStart + m == 0 || j == yBlocks - 1 && m == yLen - 1 || 
            zStart + n == 0 || k == zBlocks - 1 && n == zLen - 1)
        {
            out[l * yLen * zLen + m * zLen + n] = sin(2 * M_PI / Lx * scaleCoef * (xStart + l) * hx) * 
                sin(M_PI / Ly * scaleCoef * (yStart + m) * hy + M_PI) * 
                sin(2 * M_PI / Lz * scaleCoef * (zStart + n) * hz + 2 * M_PI) * 
                cos(M_PI * sqrt(4 / (Lx * Lx) + 1 / (Ly * Ly) + 4 / (Lz * Lz)) * stepNumber * T / K + M_PI);
        }
        else
        {
            double x0, x2, y0, y2, z0, z2, c;

            x0 = getItemGPU(in, 
                xPrev, xNext, yPrev, yNext, zPrev, zNext,
                xLen, yLen, zLen, 
                l-1, m, n);
            x2 = getItemGPU(in, 
                xPrev, xNext, yPrev, yNext, zPrev, zNext,
                xLen, yLen, zLen, 
                l+1, m, n);
            y0 = getItemGPU(in, 
                xPrev, xNext, yPrev, yNext, zPrev, zNext,
                xLen, yLen, zLen, 
                l, m-1, n);
            y2 = getItemGPU(in, 
                xPrev, xNext, yPrev, yNext, zPrev, zNext,
                xLen, yLen, zLen, 
                l, m+1, n);
            z0 = getItemGPU(in, 
                xPrev, xNext, yPrev, yNext, zPrev, zNext,
                xLen, yLen, zLen, 
                l, m, n-1);
            z2 = getItemGPU(in, 
                xPrev, xNext, yPrev, yNext, zPrev, zNext,
                xLen, yLen, zLen, 
                l, m, n+1);
            c = in[l * yLen * zLen + m * zLen + n];

            out[l * yLen * zLen + m * zLen + n] = c + (T / K) * (T / K) / 2 * (
                (x0 - 2 * c + x2) / (hx * hx) + 
                (y0 - 2 * c + y2) / (hy * hy) + 
                (z0 - 2 * c + z2) / (hz * hz)
            );

            if (l == 0 || i == 0 && l == 1)
            {
                xSelfPrev[m * zLen + n] = out[l * yLen * zLen + m * zLen + n];
            }
            if (l == xLen - 1 || i == xBlocks - 1 && l == xLen - 2)
            {
                xSelfNext[m * zLen + n] = out[l * yLen * zLen + m * zLen + n];
            }
            if (m == 0 || j == 0 && m == 1)
            {
                ySelfPrev[l * zLen + n] = out[l * yLen * zLen + m * zLen + n];
            }
            if (m == yLen - 1 || j == yBlocks - 1 && m == yLen - 2)
            {
                ySelfNext[l * zLen + n] = out[l * yLen * zLen + m * zLen + n];
            }
            if (n == 0 || k == 0 && n == 1)
            {
                zSelfPrev[l * yLen + m] = out[l * yLen * zLen + m * zLen + n];
            }
            if (n == zLen - 1 || k == zBlocks - 1 && n == zLen - 2)
            {
                zSelfNext[l * yLen + m] = out[l * yLen * zLen + m * zLen + n];
            }
        }
    }
}

__global__ void getMainIter(double *out, double *in, 
    double *xSelfPrev, double *xSelfNext, double *ySelfPrev, double *ySelfNext, double *zSelfPrev, double *zSelfNext,
    double *xPrev, double *xNext, double *yPrev, double *yNext, double *zPrev, double *zNext,
    int xLen, int yLen, int zLen,
    int xStart, int yStart, int zStart,
    int xBlocks, int yBlocks, int zBlocks,
    int i, int j, int k,
    double hx, double hy, double hz, 
    double scaleCoef, int stepNumber,
    double Lx, double Ly, double Lz,
    int K)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;

    if (l < xLen && m < yLen && n < zLen)
    {
        if (xStart + l != 0 && (i != xBlocks - 1 || l != xLen - 1) && 
            yStart + m != 0 && (j != yBlocks - 1 || m != yLen - 1) &&
            zStart + n != 0 && (k != zBlocks - 1 || n != zLen - 1))
        {
            double x0, x2, y0, y2, z0, z2, c;

            x0 = getItemGPU(in, 
                xPrev, xNext, yPrev, yNext, zPrev, zNext,
                xLen, yLen, zLen, 
                l-1, m, n);
            x2 = getItemGPU(in, 
                xPrev, xNext, yPrev, yNext, zPrev, zNext,
                xLen, yLen, zLen, 
                l+1, m, n);
            y0 = getItemGPU(in, 
                xPrev, xNext, yPrev, yNext, zPrev, zNext,
                xLen, yLen, zLen, 
                l, m-1, n);
            y2 = getItemGPU(in, 
                xPrev, xNext, yPrev, yNext, zPrev, zNext,
                xLen, yLen, zLen, 
                l, m+1, n);
            z0 = getItemGPU(in, 
                xPrev, xNext, yPrev, yNext, zPrev, zNext,
                xLen, yLen, zLen, 
                l, m, n-1);
            z2 = getItemGPU(in, 
                xPrev, xNext, yPrev, yNext, zPrev, zNext,
                xLen, yLen, zLen, 
                l, m, n+1);
            c = in[l * yLen * zLen + m * zLen + n];

            out[l * yLen * zLen + m * zLen + n] = 2 * c - out[l * yLen * zLen + m * zLen + n] + (T / K) * (T / K) * (
                (x0 - 2 * c + x2) / (hx * hx) + 
                (y0 - 2 * c + y2) / (hy * hy) + 
                (z0 - 2 * c + z2) / (hz * hz)
            );

            if (l == 0 || i == 0 && l == 1)
            {
                xSelfPrev[m * zLen + n] = out[l * yLen * zLen + m * zLen + n];
            }
            if (l == xLen - 1 || i == xBlocks - 1 && l == xLen - 2)
            {
                xSelfNext[m * zLen + n] = out[l * yLen * zLen + m * zLen + n];
            }
            if (m == 0 || j == 0 && m == 1)
            {
                ySelfPrev[l * zLen + n] = out[l * yLen * zLen + m * zLen + n];
            }
            if (m == yLen - 1 || j == yBlocks - 1 && m == yLen - 2)
            {
                ySelfNext[l * zLen + n] = out[l * yLen * zLen + m * zLen + n];
            }
            if (n == 0 || k == 0 && n == 1)
            {
                zSelfPrev[l * yLen + m] = out[l * yLen * zLen + m * zLen + n];
            }
            if (n == zLen - 1 || k == zBlocks - 1 && n == zLen - 2)
            {
                zSelfNext[l * yLen + m] = out[l * yLen * zLen + m * zLen + n];
            }
        }
    }
}

__global__ void getXLeftBorder(double *out, double *border,
    int xLen, int yLen, int zLen)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < yLen && n < zLen)
    {
        out[m * zLen + n] = (border[m * zLen + n] + out[1 * yLen * zLen + m * zLen + n]) / 2;
    }
}

__global__ void getXRightBorder(double *out, double *border,
    int xLen, int yLen, int zLen)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < yLen && n < zLen)
    {
        out[(xLen-1) * yLen * zLen + m * zLen + n] = (border[m * zLen + n] + out[(xLen-2) * yLen * zLen + m * zLen + n]) / 2;
    }
}

__global__ void getYLeftBorder(double *out,
    int xLen, int yLen, int zLen)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (l < xLen && n < zLen)
    {
        out[l * yLen * zLen + n] = 0;
    }
}

__global__ void getYRightBorder(double *out,
    int xLen, int yLen, int zLen)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (l < xLen && n < zLen)
    {
        out[l * yLen * zLen + (yLen-1) * zLen + n] = 0;
    }
}
__global__ void getZLeftBorder(double *out, double *border,
    int xLen, int yLen, int zLen)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (l < xLen && m < yLen)
    {
        out[l * yLen * zLen + m * zLen] = (border[l * yLen + m] + out[l * yLen * zLen + m * zLen + 1]) / 2;
    }
}

__global__ void getZRightBorder(double *out, double *border,
    int xLen, int yLen, int zLen)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (l < xLen && m < yLen)
    {
        out[l * yLen * zLen + m * zLen + zLen-1] = (border[l * yLen + m] + out[l * yLen * zLen + m * zLen + (zLen-2)]) / 2;
    }
}

__global__ void getErrors(double *out, double *in,
    int xLen, int yLen, int zLen, 
    int xStart, int yStart, int zStart, 
    double hx, double hy, double hz, 
    double scaleCoef, int stepNumber,
    double Lx, double Ly, double Lz,
    int K)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;

    if (l < xLen && m < yLen && n < zLen)
    {
        out[l * yLen * zLen + m * zLen + n] = fabs(in[l * yLen * zLen + m * zLen + n] - 
            sin(2 * M_PI / Lx * scaleCoef * (xStart + l) * hx) * 
            sin(M_PI / Ly * scaleCoef * (yStart + m) * hy + M_PI) * 
            sin(2 * M_PI / Lz * scaleCoef * (zStart + n) * hz + 2 * M_PI) * 
            cos(M_PI * sqrt(4 / (Lx * Lx) + 1 / (Ly * Ly) + 4 / (Lz * Lz)) * stepNumber * T / K + M_PI));
    }
}

extern "C" void copyBordersToDevice(double **borders)
{
    cudaMemcpy(dxPrev, borders[0], xBorderLength, cudaMemcpyHostToDevice);
    cudaMemcpy(dxNext, borders[1], xBorderLength, cudaMemcpyHostToDevice);
    cudaMemcpy(dyPrev, borders[2], yBorderLength, cudaMemcpyHostToDevice);
    cudaMemcpy(dyNext, borders[3], yBorderLength, cudaMemcpyHostToDevice);
    cudaMemcpy(dzPrev, borders[4], zBorderLength, cudaMemcpyHostToDevice);
    cudaMemcpy(dzNext, borders[5], zBorderLength, cudaMemcpyHostToDevice);
}

extern "C" void copyBordersToHost(double **selfBorders)
{
    cudaMemcpy(selfBorders[0], dxSelfPrev, xBorderLength, cudaMemcpyDeviceToHost);
    cudaMemcpy(selfBorders[1], dxSelfNext, xBorderLength, cudaMemcpyDeviceToHost);
    cudaMemcpy(selfBorders[2], dySelfPrev, yBorderLength, cudaMemcpyDeviceToHost);
    cudaMemcpy(selfBorders[3], dySelfNext, yBorderLength, cudaMemcpyDeviceToHost);
    cudaMemcpy(selfBorders[4], dzSelfPrev, zBorderLength, cudaMemcpyDeviceToHost);
    cudaMemcpy(selfBorders[5], dzSelfNext, zBorderLength, cudaMemcpyDeviceToHost);
}

extern "C" void runFirst(double **selfBorders,
    int xLen, int yLen, int zLen, 
    int xStart, int yStart, int zStart, 
    double hx, double hy, double hz, 
    double scaleCoef,
    double Lx, double Ly, double Lz,
    int K)
{
    matrixLength = sizeof(double) * xLen * yLen * zLen;
    xBorderLength = sizeof(double) * yLen * zLen;
    yBorderLength = sizeof(double) * xLen * zLen;
    zBorderLength = sizeof(double) * xLen * yLen;

    cudaMalloc(&dOut, matrixLength);
    cudaMalloc(&dIn, matrixLength);

    cudaMalloc(&dxPrev, xBorderLength);
    cudaMalloc(&dxNext, xBorderLength);
    cudaMalloc(&dxSelfPrev, xBorderLength);
    cudaMalloc(&dxSelfNext, xBorderLength);
    
    cudaMalloc(&dyPrev, yBorderLength);
    cudaMalloc(&dyNext, yBorderLength);
    cudaMalloc(&dySelfPrev, yBorderLength);
    cudaMalloc(&dySelfNext, yBorderLength);
    
    cudaMalloc(&dzPrev, zBorderLength);
    cudaMalloc(&dzNext, zBorderLength);
    cudaMalloc(&dzSelfPrev, zBorderLength);
    cudaMalloc(&dzSelfNext, zBorderLength);
    
    errors = (double *)malloc(matrixLength);

    dim3 blockSize(16, 8, 8); //поскольку по x получается меньше всего блоков (процессов MPI) => их размер по x больше всего
    dim3 gridSize((xLen - 1) / 16 + 1, (yLen - 1) / 8 + 1, (zLen - 1) / 8 + 1);

    getFirstIter<<<gridSize, blockSize>>>(dOut, 
        dxSelfPrev, dxSelfNext, dySelfPrev, dySelfNext, dzSelfPrev, dzSelfNext,
        xLen, yLen, zLen, 
        xStart, yStart, zStart, 
        hx, hy, hz, 
        scaleCoef,
        Lx, Ly, Lz,
        K);

    copyBordersToHost(selfBorders);
    
    double *buf = dOut;
    dOut = dIn;
    dIn = buf;
}

extern "C" void runSecond(double **selfBorders, double **borders,
    int xLen, int yLen, int zLen,
    int xStart, int yStart, int zStart,
    int xBlocks, int yBlocks, int zBlocks,
    int i, int j, int k,
    double hx, double hy, double hz, 
    double scaleCoef, int stepNumber,
    double Lx, double Ly, double Lz,
    int K)
{
    dim3 blockSize(16, 8, 8); //поскольку по x получается меньше всего блоков (процессов MPI) => их размер по x больше всего
    dim3 gridSize((xLen - 1) / 16 + 1, (yLen - 1) / 8 + 1, (zLen - 1) / 8 + 1);

    getSecondIter<<<gridSize, blockSize>>>(dOut, dIn, 
        dxSelfPrev, dxSelfNext, dySelfPrev, dySelfNext, dzSelfPrev, dzSelfNext,
        dxPrev, dxNext, dyPrev, dyNext, dzPrev, dzNext,
        xLen, yLen, zLen,
        xStart, yStart, zStart,
        xBlocks, yBlocks, zBlocks,
        i, j, k,
        hx, hy, hz, 
        scaleCoef, stepNumber,
        Lx, Ly, Lz,
        K);
        
    copyBordersToHost(selfBorders);

    double *buf = dOut;
    dOut = dIn;
    dIn = buf;
/*
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error on block (%d, %d, %d): %s\n", i, j, k, cudaGetErrorString(error));
    }
*/
}

extern "C" void runMain(double **selfBorders, double **borders,
    int xLen, int yLen, int zLen,
    int xStart, int yStart, int zStart,
    int xBlocks, int yBlocks, int zBlocks,
    int i, int j, int k,
    double hx, double hy, double hz, 
    double scaleCoef, int stepNumber,
    double Lx, double Ly, double Lz,
    int K)
{
    dim3 blockSize(16, 8, 8); //поскольку по x получается меньше всего блоков (процессов MPI) => их размер по x больше всего
    dim3 gridSize((xLen - 1) / 16 + 1, (yLen - 1) / 8 + 1, (zLen - 1) / 8 + 1);

    getMainIter<<<gridSize, blockSize>>>(dOut, dIn, 
        dxSelfPrev, dxSelfNext, dySelfPrev, dySelfNext, dzSelfPrev, dzSelfNext,
        dxPrev, dxNext, dyPrev, dyNext, dzPrev, dzNext,
        xLen, yLen, zLen, 
        xStart, yStart, zStart,
        xBlocks, yBlocks, zBlocks,
        i, j, k,
        hx, hy, hz, 
        scaleCoef, stepNumber,
        Lx, Ly, Lz,
        K);

    copyBordersToHost(selfBorders);

    double *buf = dOut;
    dOut = dIn;
    dIn = buf;
}

extern "C" void runBorders(double **borders,
    int xLen, int yLen, int zLen,
    int xBlocks, int yBlocks, int zBlocks,
    int i, int j, int k,
    int stepNumber)
{
    copyBordersToDevice(borders);

    if (stepNumber >= 2)
    {
        dim3 blockSize(32, 32);

        dim3 gridXSize((yLen - 1) / 32 + 1, (zLen - 1) / 32 + 1);

        if (i == 0)
        {
            getXLeftBorder<<<gridXSize, blockSize>>>(dIn, dxPrev, xLen, yLen, zLen);
        }
        if (i == xBlocks - 1)
        {
            getXRightBorder<<<gridXSize, blockSize>>>(dIn, dxNext, xLen, yLen, zLen);
        }

        dim3 gridYSize((xLen - 1) / 32 + 1, (zLen - 1) / 32 + 1);

        if (j == 0)
        {
            getYLeftBorder<<<gridYSize, blockSize>>>(dIn, xLen, yLen, zLen);
        }
        if (j == yBlocks - 1)
        {
            getYRightBorder<<<gridYSize, blockSize>>>(dIn, xLen, yLen, zLen);
        }
        
        dim3 gridZSize((xLen - 1) / 32 + 1, (yLen - 1) / 32 + 1);

        if (k == 0)
        {
            getZLeftBorder<<<gridZSize, blockSize>>>(dIn, dzPrev, xLen, yLen, zLen);
        }
        if (k == zBlocks - 1)
        {
            getZRightBorder<<<gridZSize, blockSize>>>(dIn, dzNext, xLen, yLen, zLen);
        }
    }    
}

extern "C" double runErrors(double *out,
    int xLen, int yLen, int zLen,
    int xStart, int yStart, int zStart,
    int i, int j, int k,
    double hx, double hy, double hz, 
    double scaleCoef, int stepNumber,
    double Lx, double Ly, double Lz,
    int K)
{
    dim3 blockSize(16, 8, 8);
    dim3 gridSize((xLen - 1) / 16 + 1, (yLen - 1) / 8 + 1, (zLen - 1) / 8 + 1);

    cudaMemcpy(out, dIn, matrixLength, cudaMemcpyDeviceToHost);

    getErrors<<<gridSize, blockSize>>>(dOut, dIn,
        xLen, yLen, zLen, 
        xStart, yStart, zStart, 
        hx, hy, hz, 
        scaleCoef, stepNumber,
        Lx, Ly, Lz,
        K);

    cudaMemcpy(errors, dOut, matrixLength, cudaMemcpyDeviceToHost);

    double error = 0.0;

    for (int l = 0; l < xLen; ++l)
    {
        for (int m = 0; m < yLen; ++m)
        {
            for (int n = 0; n < zLen; ++n)
            {
                if (errors[l * yLen * zLen + m * zLen + n] > error)
                {
                    error = errors[l * yLen * zLen + m * zLen + n];
                }
            }
        }
    }
    
    cudaFree(dOut);
    cudaFree(dIn);

    cudaFree(dxPrev);
    cudaFree(dxNext);
    cudaFree(dxSelfPrev);
    cudaFree(dxSelfNext);
    
    cudaFree(dyPrev);
    cudaFree(dyNext);
    cudaFree(dySelfPrev);
    cudaFree(dySelfNext);
    
    cudaFree(dzPrev);
    cudaFree(dzNext);
    cudaFree(dzSelfPrev);
    cudaFree(dzSelfNext);

    free(errors);

    return error;
}


