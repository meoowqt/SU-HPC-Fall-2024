#include <iostream>
#include <cuda_runtime.h>
#include "EasyBMP.h"

#define BLOCK_SIZE 16

__global__ void medianFilterKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        unsigned char window[9];
        int index = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                window[index++] = input[(y + dy) * width + (x + dx)];
            }
        }
        for (int i = 0; i < 9; i++) {
            for (int j = i + 1; j < 9; j++) {
                if (window[i] > window[j]) {
                    unsigned char temp = window[i];
                    window[i] = window[j];
                    window[j] = temp;
                }
            }
        }
        output[y * width + x] = window[4]; // Медиана - центральный элемент
    }
}

int main() {
    BMP Image;
    if (!Image.ReadFromFile("NoiseLana.bmp")) {
        std::cerr << "Ошибка: не удалось загрузить изображение." << std::endl;
        return -1;
    }

    int width = Image.TellWidth();
    int height = Image.TellHeight();

    unsigned char* h_input = new unsigned char[width * height];
    unsigned char* h_output = new unsigned char[width * height];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_input[y * width + x] = Image.GetPixel(x, y).Red;
        }
    }

    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input, h_input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    medianFilterKernel << <gridSize, blockSize >> > (d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaMemcpy(h_output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    BMP OutputImage;
    OutputImage.SetSize(width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            RGBApixel pixel;
            pixel.Red = pixel.Green = pixel.Blue = h_output[y * width + x];
            pixel.Alpha = 0;
            OutputImage.SetPixel(x, y, pixel);
        }
    }

    if (!OutputImage.WriteToFile("output.bmp")) {
        std::cerr << "Ошибка: не удалось сохранить изображение." << std::endl;
        return -1;
    }

    std::cout << "Изображение успешно обработано и сохранено." << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}