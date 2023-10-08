#include "kernel.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


#define MIN(x,y) (x > y) ? y : x;
#define TILE 16
#define TRANSPOSE_TILE 32


// General tile based linear-layer-with-bias kernel
template <unsigned TILE_SIZE>
__global__ void linearWithBiasKernel(float *z, const float *i, const float *w, const float* b, 
    const unsigned N, const unsigned D, const unsigned M)
{
    __shared__ float pulledASub[TILE_SIZE][TILE_SIZE];
    __shared__ float pulledBSub[TILE_SIZE][TILE_SIZE];


    const unsigned localY = threadIdx.x / TILE_SIZE;
    const unsigned localX = threadIdx.x % TILE_SIZE;

    const unsigned tileY = blockIdx.x * TILE_SIZE;
    const unsigned tileX = blockIdx.y * TILE_SIZE; // reversed to preserve my (and nobody else's) sanity

    float localResult = 0;
    for (unsigned tileStep = 0; tileStep < D; tileStep += TILE_SIZE) {

        const unsigned aIndY = tileY + localY;
        const unsigned aIndX = tileStep + localX;
        if (aIndY < N && aIndX < D)
            pulledASub[localY][localX] = i[(aIndY * D) + aIndX];
        else 
            pulledASub[localY][localX] = 0.0;

        const unsigned bIndY = tileStep + localY;
        const unsigned bIndX = tileX + localX;
        if (bIndY < D && bIndX < M)
            pulledBSub[localX][localY] = w[(bIndY * M) + bIndX];
        else 
            pulledBSub[localX][localY]  = 0.0;
        __syncthreads();

        #pragma unroll
        for (unsigned k = 0; k < TILE_SIZE; k++) {
            localResult += pulledASub[localY][k] * pulledBSub[localX][k];
        }
        __syncthreads();
    }

    const unsigned cIndY = tileY + localY;
    const unsigned cIndX = tileX + localX;
    if (cIndY < N && cIndX < M) {
        z[(cIndY * M) + cIndX] = localResult + b[cIndX];
    }
}

// very simple not optimized Transpose
template <unsigned TILE_SIZE>
__global__ void transposeKernel(float* mT, const float* m, unsigned N, unsigned M) {
    __shared__ float transTile[TILE_SIZE][TILE_SIZE];

    int localY = threadIdx.x / TILE_SIZE;
    int localX = threadIdx.x % TILE_SIZE;
    int row = (blockIdx.x * TILE_SIZE) + localY;
    int col = (blockIdx.y * TILE_SIZE) + localX;

    if (row < N && col < M) {
        transTile[localY][localX] = m[(row * M) + col];
    }

    localX = threadIdx.x / TILE_SIZE;
    localY = threadIdx.x % TILE_SIZE;
    row = (blockIdx.x * TILE_SIZE) + localY;
    col = (blockIdx.y * TILE_SIZE) + localX;

    __syncthreads();

    if (row < N && col < M) {
        mT[(col * N) + row] = transTile[localY][localX];
    }

}

// General tile-based mat mul kernel
template <unsigned TILE_SIZE>
__global__ void matMulKernel(float *z, const float*i, const float *w, const unsigned N, const unsigned D, const unsigned M) {
    __shared__ float pulledASub[TILE_SIZE][TILE_SIZE];
    __shared__ float pulledBSub[TILE_SIZE][TILE_SIZE];


    const unsigned localY = threadIdx.x / TILE_SIZE;
    const unsigned localX = threadIdx.x % TILE_SIZE;

    const unsigned tileY = blockIdx.x * TILE_SIZE;
    const unsigned tileX = blockIdx.y * TILE_SIZE; // reversed to preserve my sanity

    double localResult = 0;
    for (unsigned tileStep = 0; tileStep < D; tileStep += TILE_SIZE) {
        const unsigned aIndY = tileY + localY;
        const unsigned aIndX = tileStep + localX;
        if (aIndY < N && aIndX < D) {
            pulledASub[localY][localX] = i[(aIndY * D) + aIndX];
        } else {
            pulledASub[localY][localX] = 0.0;
        }
        const unsigned bIndY = tileStep + localY;
        const unsigned bIndX = tileX + localX;
        if (bIndY < D && bIndX < M) {
            pulledBSub[localX][localY] = w[(bIndY * M) + bIndX];
        } else {
            pulledBSub[localX][localY]  = 0.0; // TRANSPOSED HERE
        }
        __syncthreads();

        #pragma unroll
        for (unsigned k = 0; k < TILE_SIZE; k++) {
            localResult += (pulledASub[localY][k] * pulledBSub[localX][k]);
        }
        __syncthreads();
    }

    const unsigned cIndY = tileY + localY;
    const unsigned cIndX = tileX + localX;
    if (cIndY < N && cIndX < M) {
        z[(cIndY * M) + cIndX] = localResult;
    }

}


// very simple not optimized sum. Each block takes a chunk of 32 columns, each warp divides the columns into rows
__global__ void sumFirstDimension(float *sum, const float* m, const unsigned N, const unsigned M) {
    __shared__ float runningSum[32];

    unsigned blockX = blockIdx.x * 32;

    unsigned localY = threadIdx.x / 32;
    unsigned localX = threadIdx.x % 32;

    if (localY == 0) {
        runningSum[localX] = 0;
    }
    __syncthreads();

    unsigned linesHandledByBlock = blockDim.x / 32;
    unsigned U_warp = (N + linesHandledByBlock - 1) / linesHandledByBlock;

    unsigned startY = U_warp * localY;
    unsigned endY = MIN(U_warp * (localY + 1), N);
    unsigned x = blockX + localX;

    if (x < M) {
        float partialSum = 0;
        for (unsigned y = startY; y < endY; y++) {
            partialSum += m[(y * M) + x];
        }
        atomicAdd(&runningSum[localX], partialSum);
    }
    
    __syncthreads();

    if (localY == 0 && x < M) {
        sum[x] = runningSum[localX];
    }
}

__global__ void dummydevice() {
    
}

void customlinear(array2d_t<float>& input, array2d_t<float>& weight, array1d_t<float>& bias, array2d_t<float>& result){

    const unsigned numXBlocks = (result.col_count + TILE - 1) / TILE;
    const unsigned numYBlocks = (result.row_count + TILE - 1) / TILE;
    dim3 dimGrid(numYBlocks, numXBlocks); // NOTE: variable-wise this reverses the x and y, but that is never done functionally...

    linearWithBiasKernel<TILE> <<<dimGrid, TILE*TILE>>> (result.data_ptr, input.data_ptr, weight.data_ptr, bias.data_ptr, 
                                                         input.row_count, input.col_count, weight.col_count);
    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
    }
}


void customlinear_back(array2d_t<float>& dZ, array2d_t<float>& input, array2d_t<float>& weight, array2d_t<float>& dX, array2d_t<float>& dW, array1d_t<float>& dB){
    cudaDeviceSynchronize();

    float* transI;
    float* transW;
    cudaMalloc((void**)&transI, input.row_count * input.col_count * sizeof(float));
    cudaMalloc((void**)&transW, weight.row_count * weight.col_count * sizeof(float));

    // dW
    unsigned numXBlocks = (input.col_count + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE;
    unsigned numYBlocks = (input.row_count + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE;
    dim3 dimGrid(numYBlocks, numXBlocks);
    transposeKernel<TRANSPOSE_TILE><<<dimGrid, TRANSPOSE_TILE*TRANSPOSE_TILE>>>(transI, input.data_ptr, input.row_count, input.col_count);

    numXBlocks = (dW.col_count + TILE - 1) / TILE;
    numYBlocks = (dW.row_count + TILE - 1) / TILE;
    dimGrid.x = numYBlocks; dimGrid.y = numXBlocks;
    matMulKernel<TILE><<<dimGrid, TILE*TILE>>>(dW.data_ptr, transI, dZ.data_ptr, input.col_count, input.row_count, dZ.col_count);

    // dX
    numXBlocks = (weight.col_count + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE;
    numYBlocks = (weight.row_count + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE;
    dimGrid.x = numYBlocks; dimGrid.y = numXBlocks;
    transposeKernel<TRANSPOSE_TILE><<<dimGrid, TRANSPOSE_TILE*TRANSPOSE_TILE>>>(transW, weight.data_ptr, weight.row_count, weight.col_count);

    numXBlocks = (dX.col_count + TILE - 1) / TILE;
    numYBlocks = (dX.row_count + TILE - 1) / TILE;
    dimGrid.x = numYBlocks; dimGrid.y = numXBlocks;
    matMulKernel<TILE><<<dimGrid, TILE*TILE>>>(dX.data_ptr, dZ.data_ptr, transW, dZ.row_count, dZ.col_count, weight.row_count);

    // dB
    unsigned numBlocks = (dZ.col_count + 31) / 32;
    sumFirstDimension<<<numBlocks, 512>>>(dB.data_ptr, dZ.data_ptr, dZ.row_count, dZ.col_count);

    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
    }

    cudaFree(transI);
    cudaFree(transW);
}    







/*****************************************************************************************

******************************************************************************************/




void confirm_transpose(array2d_t<float>& a, array2d_t<float>& aT, const char* msg) {
    for (unsigned y = 0; y < a.row_count; ++y) {
        for (unsigned x = 0; x < a.col_count; ++x) {
            float diff = a.data_ptr[(y * a.col_count) + x] - aT.data_ptr[(x * a.row_count) + y];
            if (diff > 0.01 || diff < -0.01) {
                printf("Definitely failed: %d, %d for %s\n", y, x, msg);
                return;
            }
        }
    }
}


array2d_t<float> confirm_matmul(array2d_t<float>& a, array2d_t<float>& b, array2d_t<float>& r, const char * msg) {

    if (a.col_count != b.row_count) {
        return a;
    }

    array2d_t<float> compRes{new float[a.row_count * b.col_count], a.row_count, b.col_count};

    for (int i = 0; i < a.row_count; ++i) {
        for (int j = 0; j < b.col_count; ++j) {
            double sum = 0.0f;
            for (int k = 0; k < a.col_count; ++k) {
                sum += a.data_ptr[i * a.col_count + k] * b.data_ptr[k * b.col_count + j];
            }
            compRes.data_ptr[i * compRes.col_count + j] = sum;
        }
    }

    ///
    for (unsigned y = 0; y < r.row_count; ++y) {
        for (unsigned x = 0; x < r.col_count; ++x) {
            float diff = r.data_ptr[(y * r.col_count) + x] - compRes.data_ptr[(x * compRes.row_count) + y];
            if (diff > 0.01 || diff < -0.01) {
                printf("MATMUL failed: %d, %d for %s, %f\n", y, x, msg, diff);
                return compRes;
            }
        }
    }

    return compRes;

}