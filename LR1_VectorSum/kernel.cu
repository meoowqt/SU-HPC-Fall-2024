#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

__global__ void sumArray(int* input, int* output, int size) {
    extern __shared__ int sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        sharedData[tid] = input[i];
    }
    else {
        sharedData[tid] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int sumArrayCPU(int* input, int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += input[i];
    }
    return sum;
}

int sumArrayGPU(int* input, int size) {
    int* d_input, * d_output;
    cudaMalloc((void**)&d_input, size * sizeof(int));

    int blockSize = 1024;
    int gridSize = (size + blockSize - 1) / blockSize;
    cudaMalloc((void**)&d_output, gridSize * sizeof(int));

    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    sumArray << <gridSize, blockSize, blockSize * sizeof(int) >> > (d_input, d_output, size);
    cudaDeviceSynchronize(); 

    int* h_output = new int[gridSize];
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    int finalSumGPU = 0;
    for (int i = 0; i < gridSize; ++i) {
        finalSumGPU += h_output[i];
    }
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;

    return finalSumGPU;
}

int main() {
    std::vector<int> arraySizes = { 1000, 10000, 100000, 1000000, 10000000, 100000000 };

    for (int size : arraySizes) {
        int* h_input = new int[size];

        for (int i = 0; i < size; ++i) {
            h_input[i] = 1;
        }

        auto startGPU = std::chrono::high_resolution_clock::now();
        int finalSumGPU = sumArrayGPU(h_input, size);
        auto endGPU = std::chrono::high_resolution_clock::now();

        auto startCPU = std::chrono::high_resolution_clock::now();
        int finalSumCPU = sumArrayCPU(h_input, size);
        auto endCPU = std::chrono::high_resolution_clock::now();

        std::cout << "Array size: " << size << std::endl;
        std::cout << "Sum of array elements on GPU: " << finalSumGPU << std::endl;
        std::cout << "Sum of array elements on CPU: " << finalSumCPU << std::endl;

        std::chrono::duration<double> durationGPU = endGPU - startGPU;
        std::chrono::duration<double> durationCPU = endCPU - startCPU;

        std::cout << "Execution time on GPU: " << durationGPU.count() << " seconds" << std::endl;
        std::cout << "Execution time on CPU: " << durationCPU.count() << " seconds" << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        delete[] h_input;
    }

    return 0;
}
