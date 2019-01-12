//#include <iostream>
#include <cmath>

const double PI = 3.141592653589793238463;

void generateKernel(int width, int height, double sigma, double *kernel)
{
	double sum = 0.0;
	int kernelWidth = width / 2;
	int kernelHeight = height / 2;
	
	// generate kernel
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			kernel[(i * width) + j] = exp(-(pow(i - kernelHeight, 2) + pow(j - kernelWidth, 2)) / (2 * sigma * sigma))
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
