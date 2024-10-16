#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

#define N1 100 // Размер матрицы
#define N2 2000 // Размер матрицы

// Функция для перемножения матриц на CPU
vector<int> multiplyCPU(const vector<int>& A, const vector<int>& B, int n) {
    vector<int> C(n * n, 0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }

    return C;
}

// CUDA Kernel для перемножения матриц на GPU
__global__ void multiplyGPU(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < n && col < n) {
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Функция для перемножения матриц на GPU
vector<int> multiplyMatrixOnGPU(const vector<int>& A, const vector<int>& B, int n) {
    int size = n * n * sizeof(int);

    // Аллоцирование памяти на GPU
    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Копирование данных из CPU в GPU (одномерные массивы)
    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    // Определение конфигурации блоков и потоков
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Запуск Kernel
    multiplyGPU <<< blocksPerGrid, threadsPerBlock >>> (d_A, d_B, d_C, n);

    // Копирование результата обратно в CPU
    vector<int> result(n * n);
    cudaMemcpy(result.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}

int main() {
    int n1 = N1;
    int n2 = N2;
    // Используем одномерные вектора для представления матриц
    for (int n = N1; n <= 2000; n += 500) {
        vector<int> A(n * n, 1);
        vector<int> B(n * n, 1);

        // Время выполнения на GPU
        auto startGPU = chrono::high_resolution_clock::now();
        vector<int> C_GPU = multiplyMatrixOnGPU(A, B, n);
        auto endGPU = chrono::high_resolution_clock::now();
        chrono::duration<double> durationGPU = endGPU - startGPU;
        cout << "Execution time on GPU for " << n << "x" << n << ": " << durationGPU.count() << " sec\n";

        // Время выполнения на CPU
        auto startCPU = chrono::high_resolution_clock::now();
        vector<int> C_CPU = multiplyCPU(A, B, n);
        auto endCPU = chrono::high_resolution_clock::now();
        chrono::duration<double> durationCPU = endCPU - startCPU;
        cout << "Execution time on CPU for " << n << "x" << n << ": " << durationCPU.count() << " sec\n";
        
        // Проверка на правильность вычисления
        bool correct = true;
        for (int i = 0; i < n * n; ++i) {
            if (C_CPU[i] != C_GPU[i]) {
                correct = false;
                break;
            }
        }

        if (correct) {
            cout << "Matrix multiplication is correct.\n";
        }
        else {
            cout << "Matrix multiplication is incorrect.\n";
        }

        cout << "\n";
         
    }
    return 0;
}