#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include "EasyBMP.h"

#define BLOCK_SIZE 1

void loadImage(const std::string& filename, std::vector<unsigned char>& image, int& width, int& height) {
    BMP Input;
    Input.ReadFromFile(filename.c_str());
    width = Input.TellWidth();
    height = Input.TellHeight();

    image.resize(width * height);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            image[i * width + j] = Input(j, i)->Red;
        }
    }
}

void saveImage(const std::string& filename, const std::vector<unsigned char>& image, const std::vector<unsigned char>& corners, int width, int height) {
    BMP Output;
    Output.SetSize(width, height);
    Output.SetBitDepth(24);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            if (corners[idx] == 255) {
                Output(j, i)->Red = 255;
                Output(j, i)->Green = 0;
                Output(j, i)->Blue = 0;
            }
            else {
                Output(j, i)->Red = image[idx];
                Output(j, i)->Green = image[idx];
                Output(j, i)->Blue = image[idx];
            }
        }
    }
    Output.WriteToFile(filename.c_str());
}

__global__ void computeGradients(unsigned char* input, float* Ix, float* Iy, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 0 && x < width && y >= 0 && y < height) {
        int idx = y * width + x;
        if (x > 0 && x < width - 1) {
            Ix[idx] = (input[idx + 1] - input[idx - 1]) / 2.0f;
        }
        if (y > 0 && y < height - 1) {
            Iy[idx] = (input[idx + width] - input[idx - width]) / 2.0f;
        }
    }
}

__global__ void computeHarrisMatrix(float* Ix, float* Iy, float* A11, float* A12, float* A22, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 0 && x < width && y >= 0 && y < height) {
        int idx = y * width + x;
        A11[idx] = Ix[idx] * Ix[idx];
        A12[idx] = Ix[idx] * Iy[idx];
        A22[idx] = Iy[idx] * Iy[idx];
    }
}

__global__ void applyGaussianBlur(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 0 && x < width && y >= 0 && y < height) {
        int idx = y * width + x;
        float sum = 0.0f;
        int count = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[ny * width + nx];
                    count++;
                }
            }
        }
        output[idx] = sum / count;
    }
}

__global__ void computeHarrisResponse(float* A11, float* A12, float* A22, float* R, int width, int height, float k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 0 && x < width && y >= 0 && y < height) {
        int idx = y * width + x;
        float det = A11[idx] * A22[idx] - A12[idx] * A12[idx];
        float trace = A11[idx] + A22[idx];
        R[idx] = det - k * trace * trace;
    }
}

__global__ void markCorners(float* R, unsigned char* output, int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 0 && x < width && y >= 0 && y < height) {
        int idx = y * width + x;

        if (R[idx] > threshold) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int neighborIdx = ny * width + nx;
                        output[neighborIdx] = 255;
                    }
                }
            }
        }
    }
}


int main() {
    std::string inputFilename = "input.bmp";
    std::string outputFilename = "output.bmp";
    float threshold = 10000.0f;
    float k = 0.04f;

    int width, height;
    std::vector<unsigned char> image;
    loadImage(inputFilename, image, width, height);

    unsigned char* d_input;
    float* d_Ix, * d_Iy, * d_A11, * d_A12, * d_A22, * d_R;
    unsigned char* d_output;

    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_Ix, width * height * sizeof(float));
    cudaMalloc(&d_Iy, width * height * sizeof(float));
    cudaMalloc(&d_A11, width * height * sizeof(float));
    cudaMalloc(&d_A12, width * height * sizeof(float));
    cudaMalloc(&d_A22, width * height * sizeof(float));
    cudaMalloc(&d_R, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input, image.data(), width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    computeGradients << <gridSize, blockSize >> > (d_input, d_Ix, d_Iy, width, height);
    computeHarrisMatrix << <gridSize, blockSize >> > (d_Ix, d_Iy, d_A11, d_A12, d_A22, width, height);

    float* d_A11_blur, * d_A12_blur, * d_A22_blur;
    cudaMalloc(&d_A11_blur, width * height * sizeof(float));
    cudaMalloc(&d_A12_blur, width * height * sizeof(float));
    cudaMalloc(&d_A22_blur, width * height * sizeof(float));

    applyGaussianBlur << <gridSize, blockSize >> > (d_A11, d_A11_blur, width, height);
    applyGaussianBlur << <gridSize, blockSize >> > (d_A12, d_A12_blur, width, height);
    applyGaussianBlur << <gridSize, blockSize >> > (d_A22, d_A22_blur, width, height);

    computeHarrisResponse << <gridSize, blockSize >> > (d_A11_blur, d_A12_blur, d_A22_blur, d_R, width, height, k);
    markCorners << <gridSize, blockSize >> > (d_R, d_output, width, height, threshold);

    std::vector<unsigned char> outputImage(width * height);
    cudaMemcpy(outputImage.data(), d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    saveImage(outputFilename, image, outputImage, width, height);

    cudaFree(d_input);
    cudaFree(d_Ix);
    cudaFree(d_Iy);
    cudaFree(d_A11);
    cudaFree(d_A12);
    cudaFree(d_A22);
    cudaFree(d_R);
    cudaFree(d_A11_blur);
    cudaFree(d_A12_blur);
    cudaFree(d_A22_blur);
    cudaFree(d_output);

    return 0;
}