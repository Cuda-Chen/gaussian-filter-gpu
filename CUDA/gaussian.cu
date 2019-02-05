#include <cmath>
#include <iostream>

#ifdef defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	#define WINDOWS_LEAN_AND_MEAN
	#define NOMINMAX
	#include <windows.h>
#endif

#include "cuda.h"
#include "cuda_runtime.h"

#include "gaussian.hpp"

using std::cerr;
using std::endl;

const double PI = 3.14159;

inline void checkError(cudaError_t status)
{
	if(status != cudaSuccess)
	{
		cerr << "Cuda failure "
			<< __FILE__ << ":"
			<< __LINE__ << ": "
			<< cudaGetErrorString(status) << endl;
	}
}

void generateKernel(int width, int height, double sigma, double *kernel)
{
	double sum = 0.0;
	int strideWidth = width / 2;
	int strideHeight = height / 2;
	
	// generate kernel
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			kernel[(i * width) + j] = exp(-(pow(i - strideHeight, 2) + pow(j - strideWidth, 2)) / (2 * sigma * sigma))
				/ (2 * PI * sigma * sigma);
			sum += kernel[(i * width) + j];
		}
	}

	// then normalize each element
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			kernel[(i * width) + j] /= sum;
		}
	}
}

__global__ void gaussian(unsigned char *src, unsigned char *dst,
	double *gaussianKernel, int width,
	int height, int kernelWidth,
	int kernelHeight, double sigma)
{
	int strideWidth = kernelWidth / 2;
	int strideHeight = kernelHeight / 2;

	int row = blockIdx.x * blockDim.x + threadIdx.x + strideHeight;
	int col = blockIdx.y * blockDim.y + threadIdx.y + strideWidth;

	// boundary check
	if(row < 0 || col < 0 || row > height || col > width)
	{
		return;
	}

	double temp = 0.0;
	int xindex, yindex;

	for(int krow = 0; krow < kernelHeight; krow++)
	{
		for(int kcol = 0; kcol < kernelWidth; kcol++)
		{
			xindex = krow + row - strideHeight;
			yindex = kcol + col - strideWidth;
			temp += src[(xindex * width) + yindex] * gaussianKernel[(krow * kernelWidth) + kcol];
		}
	}
	__syncthreads();

	if(temp > 255)
	{
		temp = 255;
	}
	else if(temp < 0)
	{
		temp = 0;
	}

	dst[(row * width) + col] = (unsigned char)temp;
}

void gaussianFilter(unsigned char *src, unsigned char *dst,
	int width, int height, int kernelWidth, int kernelHeight, double sigma)
{
	double *kernel = new double[kernelWidth * kernelHeight];

	generateKernel(kernelWidth, kernelHeight, sigma, kernel);

	const dim3 blockSize(16, 16, 1);
	const dim3 gridSize(width / blockSize.x + 1, height / blockSize.y + 1, 1);

	unsigned char *d_src, *d_dst;
	double *d_kernel;

	checkError(cudaMalloc(&d_src, height * width * sizeof(unsigned char)));
	checkError(cudaMalloc(&d_dst, height * width * sizeof(unsigned char)));
	checkError(cudaMalloc(&d_kernel, kernelHeight * kernelWidth * sizeof(double)));

	checkError(cudaMemcpy(d_src, src,
		height * width * sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkError(cudaMemcpy(d_dst, dst,
		height * width * sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkError(cudaMemcpy(d_kernel, kernel,
		kernelHeight * kernelWidth * sizeof(double), cudaMemcpyHostToDevice));

	gaussian<<<1, 1>>>(d_src, d_dst, d_kernel, width,
		height, kernelWidth, kernelHeight, sigma);

	checkError(cudaMemcpy(dst, d_dst,
		height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// release CUDA objects
	cudaFree(d_src);
	cudaFree(d_dst);
	cudaFree(d_kernel);

	// release host objects
	delete [] kernel;
}

