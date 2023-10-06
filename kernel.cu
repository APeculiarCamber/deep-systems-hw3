#include "kernel.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

template <typename T, unsigned TILE_SIZE>
__global__ void linearWithBiasKernel(T *z, const T *i, const T *w, const T* b, 
    const unsigned N, const unsigned D, const unsigned M)
{
    __shared__ float pulledASub[TILE_SIZE][TILE_SIZE];
    __shared__ float pulledBSub[TILE_SIZE][TILE_SIZE];


    const unsigned localY = threadIdx.x / TILE_SIZE;
    const unsigned localX = threadIdx.x % TILE_SIZE;

    const unsigned tileY = blockIdx.x * TILE_SIZE;
    const unsigned tileX = blockIdx.y * TILE_SIZE; // reversed to preserve my (and nobody else's) sanity

    float localResult = 0;
    unsigned aTileX = 0;
    // aTileY = tileY;
    unsigned bTileY = 0;
    // bTileX = tileX;
    for (; aTileX < D; aTileX += TILE_SIZE, bTileY += TILE_SIZE) {

        const unsigned aIndY = tileY + localY;
        const unsigned aIndX = aTileX + localX;
        if (aIndY < N && aIndX < D)
            pulledASub[localY][localX] = i[(aIndY * D) + aIndX];
        else 
            pulledASub[localY][localX] = 0.0;

        const unsigned bIndY = bTileY + localY;
        const unsigned bIndX = tileX + localX;
        if (bIndY < D && bIndX < M)
            pulledBSub[localX][localY] = w[(bIndY * M) + bIndX];
        else 
            pulledBSub[localX][localY]  = 0.0; // TRANSPOSED HERE! VERIFY SPEED AND EFFECT
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

template <typename T, unsigned TILE_SIZE>
__global__ void transKernel(T* m, T* mT, unsigned N, unsigned M) {
    // TODO
    // 🏳️‍⚧️

}

template <typename T, unsigned TILE_SIZE>
__global__ void matMulKernel(T *c, const T *a, const T *b, const unsigned N, const unsigned D, const unsigned M) {
__shared__ float pulledASub[TILE_SIZE][TILE_SIZE];
    __shared__ float pulledBSub[TILE_SIZE][TILE_SIZE];


    const unsigned localY = threadIdx.x / TILE_SIZE;
    const unsigned localX = threadIdx.x % TILE_SIZE;

    const unsigned tileY = blockIdx.x * TILE_SIZE;
    const unsigned tileX = blockIdx.y * TILE_SIZE; // reversed to preserve my (and nobody else's) sanity

    float localResult = 0;
    unsigned aTileX = 0;
    // aTileY = tileY;
    unsigned bTileY = 0;
    // bTileX = tileX;
    for (; aTileX < D; aTileX += TILE_SIZE, bTileY += TILE_SIZE) {

        const unsigned aIndY = tileY + localY;
        const unsigned aIndX = aTileX + localX;
        if (aIndY < N && aIndX < D)
            pulledASub[localY][localX] = a[(aIndY * D) + aIndX];
        else 
            pulledASub[localY][localX] = 0.0;

        const unsigned bIndY = bTileY + localY;
        const unsigned bIndX = tileX + localX;
        if (bIndY < D && bIndX < M)
            pulledBSub[localX][localY] = b[(bIndY * M) + bIndX];
        else 
            pulledBSub[localX][localY]  = 0.0; // TRANSPOSED HERE! VERIFY SPEED AND EFFECT
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
        c[(cIndY * M) + cIndX] = localResult + b[cIndX];
    }
}

template <typename T>
__global__ void sumFirstDimension(T *sum, T* m, const unsigned N, const unsigned M) {
    // TODO
}

#define TILE 16

void customlinear(array2d_t<float>& input, array2d_t<float>& weight, array1d_t<float>& bias, array2d_t<float>& result){
    // ASSUMING ROW MAJOR!
    const unsigned numXBlocks = (result.col_count + TILE - 1) / TILE;
    const unsigned numYBlocks = (result.row_count + TILE - 1) / TILE;
    dim3 dimGrid(numYBlocks, numXBlocks); // NOTE: variables will reverse the x and y, but we never do functionally...
    // printf("Running with <<<(%d, %d, %d), %d>>>\n", dimGrid.x, dimGrid.y, dimGrid.z, TILE * TILE);
    linearWithBiasKernel<float, TILE><<<dimGrid, TILE*TILE>>> (result.data_ptr, input.data_ptr, weight.data_ptr, bias.data_ptr, 
                                                        input.row_count, input.col_count, weight.col_count);
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error or exit the program.
    }
}

void customlinear_back(array2d_t<float>& dZ, array2d_t<float>& input, array2d_t<float>& weight, array2d_t<float>& dX, array2d_t<float>& dW, array1d_t<float>& dB){;
    // boring transpose into boring mat mul
    float* transI;
    float* transW;
    cudaMalloc((void**)&transI, input.row_count * input.col_count * sizeof(float));
    cudaMalloc((void**)&transW, weight.row_count * weight.col_count * sizeof(float));

    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    cudaStreamCreate(&stream[2]);
    // boring sum up for bias

    // dW
    unsigned numXBlocks = (input.col_count + TILE - 1) / TILE;
    unsigned numYBlocks = (input.row_count + TILE - 1) / TILE;
    dim3 dimGrid(numYBlocks, numXBlocks);
    transKernel<float, TILE><<<dimGrid, TILE*TILE, 0, stream[0]>>>(input.data_ptr, transI, input.row_count, input.col_count);
    numXBlocks = (dW.col_count + TILE - 1) / TILE;
    numYBlocks = (dW.row_count + TILE - 1) / TILE;
    dimGrid.x = numYBlocks; dimGrid.y = numXBlocks;
    matMulKernel<float, TILE><<<dimGrid, TILE, 0, stream[0]>>>(dW.data_ptr, transI, dZ.data_ptr, input.col_count, input.row_count, dZ.col_count);

    // dX
    numXBlocks = (weight.col_count + TILE - 1) / TILE;
    numYBlocks = (weight.row_count + TILE - 1) / TILE;
    dimGrid.x = numYBlocks; dimGrid.y = numXBlocks;
    transKernel<float, TILE><<<dimGrid, TILE*TILE, 0, stream[1]>>>(weight.data_ptr, transW, input.row_count, input.col_count);
    numXBlocks = (dX.col_count + TILE - 1) / TILE;
    numYBlocks = (dX.row_count + TILE - 1) / TILE;
    dimGrid.x = numYBlocks; dimGrid.y = numXBlocks;
    matMulKernel<float, TILE><<<dimGrid, TILE, 0, stream[1]>>>(dX.data_ptr, dZ.data_ptr, transW, dZ.row_count, dZ.col_count, weight.row_count);

    // dB
    unsigned numBlocks = (dZ.col_count + 511) / 512;
    sumFirstDimension<float><<<numBlocks, 512, 0, stream[2]>>>(dB.data_ptr, dZ.data_ptr, dZ.row_count, dZ.col_count);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);
    cudaStreamSynchronize(stream[2]);

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
}    


/*T* dev_a = 0;
    T* dev_b = 0;
    T* dev_c = 0;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0); 
    cudaStatus = cudaMalloc((void**)&dev_a, a.row_count * a.col_count * sizeof(T));
    cudaStatus = cudaMalloc((void**)&dev_b, b.row_count * b.col_count * sizeof(T));
    cudaStatus = cudaMalloc((void**)&dev_c, c.row_count * c.col_count * sizeof(T));
    cudaStatus = cudaMemcpy(dev_a, a.data_ptr, a.row_count * a.col_count * sizeof(T), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b.data_ptr, b.row_count * b.col_count * sizeof(T), cudaMemcpyHostToDevice);*/
