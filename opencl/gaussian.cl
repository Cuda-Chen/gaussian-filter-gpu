__kernel void gaussian(__global unsigned char *src, 
	__global unsigned char *dst, 
	__global double *gaussianKernel, 
	int width, 
	int height, 
	int kernelWidth, 
	int kernelHeight, 
	double sigma)
{
	int strideWidth = kernelWidth / 2;
	int strideHeight = kernelHeight / 2;

	unsigned int row = get_global_id(0) + strideHeight;
	unsigned int col = get_global_id(1) + strideWidth;

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
