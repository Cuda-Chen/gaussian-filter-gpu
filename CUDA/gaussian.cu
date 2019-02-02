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

void gaussianFilter(unsigned char *src, unsigned char *dst,
	int width, int height, int kernelWidth, int kernelHeight, double sigma)
{
	double *kernel = new double[kernelWidth * kernelHeight];

	generateKernel(kernelWidth, kernelHeight, sigma, kernel);

	const dim3 blockSize(16, 16, 1)
	const dim3 gridSize(width / blockSize.x + 1, height / blockSize.y + 1, 1);

	gaussian<<<gridSize, blockSize>>>(src, dst, kernel, width,
		height, kernelWidth, kernelHeight, sigma);

	/*
	int strideWidth = kernelWidth / 2;
	int strideHeight = kernelHeight / 2;

	for(int row = 0 + strideHeight; row < height - strideHeight; row++)
	{
		for(int col = 0 + strideWidth; col < width - strideWidth; col++)
		{
			double temp = 0.0;
			int xindex;
			int yindex;
			
			for(int krow = 0; krow < kernelHeight; krow++)
			{
				for(int kcol = 0; kcol < kernelWidth; kcol++)
				{
					xindex = krow + row - strideHeight;
					yindex = kcol + col - strideWidth;
					temp += src[(xindex * width) + yindex] * kernel[(krow * kernelWidth) + kcol];
				}
			}

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
	} */

	delete [] kernel;
}

__global__ void gaussian(unsigned char *src, unsigned char *dst
	, double *gaussianKernel, int width
	, int height, int kernelWidth,
	, int kernelHeight, double sigma)
{
	int strideWidth = kernelWidth / 2;
	int strideHeight = kernelHeight / 2;

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	// boundary check
	if(row < 0 || col < 0 || row > height || col > width)
	{
		return;
	}

	double temp = 0.0;
	int xindex, yindex;

	for(int krow = 0; krow < kernelHeight, krow++)
	{
		for(int kcol = 0; kcol < kernelWidth; kcol++)
		{
			xindex = krow + row - strideHeight;
			yindex = kcol + col - strideWidth;
			temp += src[(xindex * width) + yindex] * gaussianKernel[(krow * kernelWidth) + kcol];
		}
	}

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
