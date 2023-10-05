
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

template <typename T>
struct scalar2d {
    T* data_ptr;
    unsigned row, col;
};

template<typename T>
cudaError_t mat_mul(scalar2d<T>& a, scalar2d<T>& b, scalar2d<T>& c);

template <typename T, unsigned TILE_SIZE>
__global__ void matMulKernel(T *c, const T *a, const T *b, 
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
        //c[(cIndY * M) + cIndX] = localResult;
    }


    // easy to do naively...
    // TILING: for warp-sized-handle-able block in a, keep running down transposed-sized blocks in b
        // I think these calculation blocks can be any shape, so long as they eventually run the full tile-space (extended from tile to bounds)
    // There's... complications everywhere tho...


    // TILE_X x TILE_Y (like 64 to be 8 x 8?)
        // Assigned per block
        // Each tile assigned evenly per warp

    // TODO : tiling is less complicated and cooler than initially thought...
        // Bi-step grab of blocks, then we bite them both off by block
        // How is this effective however? It shouldn't be... I dont think...?
}


template<typename T>
cudaError_t mat_mul(scalar2d<T>& a, scalar2d<T>& b, scalar2d<T>& c) {
    T* dev_a = 0;
    T* dev_b = 0;
    T* dev_c = 0;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    cudaStatus = cudaMalloc((void**)&dev_a, a.row * a.col * sizeof(T));
    cudaStatus = cudaMalloc((void**)&dev_b, b.row * b.col * sizeof(T));
    cudaStatus = cudaMalloc((void**)&dev_c, c.row * c.col * sizeof(T));
    cudaStatus = cudaMemcpy(dev_a, a.data_ptr, a.row * a.col * sizeof(T), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b.data_ptr, b.row * b.col * sizeof(T), cudaMemcpyHostToDevice);

    // ASSUMING ROW MAJOR!
    #define TILE 16
    const unsigned numXBlocks = (c.col + TILE - 1) / TILE;
    const unsigned numYBlocks = (c.row + TILE - 1) / TILE;
    dim3 dimGrid(numYBlocks, numXBlocks); // NOTE: variables will reverse the x and y, but we never do functionally...
    printf("Running with <<<(%d, %d, %d), %d>>>\n", dimGrid.x, dimGrid.y, dimGrid.z, TILE * TILE);
    matMulKernel<T, TILE><<<dimGrid, TILE*TILE>>> (dev_c, dev_a, dev_b, a.row, a.col, b.col);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error or exit the program.
    }

    cudaStatus = cudaMemcpy(c.data_ptr, dev_c, c.row * c.col * sizeof(T), cudaMemcpyDeviceToHost);

    cudaStatus = cudaDeviceSynchronize();
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

template <typename T, unsigned N, unsigned D, unsigned M>
void init(scalar2d<T>& a, scalar2d<T>& b, scalar2d<T>& c) {
    a.row = N;
    a.col = D;
    b.row = D;
    b.col = M;
    c.row = a.row;
    c.col = b.col;

    a.data_ptr = new T[N * D];
    b.data_ptr = new T[D * M];
    c.data_ptr = new T[N * M];
    for (unsigned i = 0; i < N * D; i++) {
        a.data_ptr[i] = (T)i;
    }
    for (unsigned i = 0; i < M * D; i++) {
        b.data_ptr[i] = (T)i;
    }
}

template <typename T>
void eval(scalar2d<T>& a, scalar2d<T>& b, scalar2d<T>& c) {
    scalar2d<T> tempC;
    tempC.row = a.row;
    tempC.col = b.col;
    tempC.data_ptr = new float[a.row * b.col];

    // Perform matrix multiplication
    for (int i = 0; i < a.row; i++) {
        for (int j = 0; j < b.col; j++) {
            for (int k = 0; k < a.col; k++) {
                tempC.data_ptr[i * c.col + j] += a.data_ptr[i * a.col + k] * b.data_ptr[k * b.col + j];
            }
        }
    }

    float totalAfter = 0.0;
    for (int i = 0; i < c.row; i++) {
        for (int j = 0; j < c.col; j++) {
            // printf("%0.2f ", tempC.data_ptr[i * c.col + j] - c.data_ptr[i * c.col + j]);
            totalAfter += tempC.data_ptr[i * c.col + j] - c.data_ptr[i * c.col + j];
        }
        // printf("%1.1f\n");
    }
    printf("Final Diff Sum: %f\n", totalAfter);
    delete[] tempC.data_ptr;
}

int main()
{
    scalar2d<float> a, b, c;
    init<float, 52, 48, 27>(a, b, c);

    // Add vectors in parallel.
    cudaError_t cudaStatus = mat_mul<float>(c, a, b);

    eval(a, b, c);

    delete[] a.data_ptr;
    delete[] b.data_ptr;
    delete[] c.data_ptr;

    return 0;
}