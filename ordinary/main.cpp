#include <iostream>

#include "opencv2/opencv.hpp"

#include "gaussian.hpp"

int main(int argc, char **argv)
{
	using namespace std;
	using namespace cv;

	//cout << "Gaussian filter ordinary version under construction" << endl;
	Mat img, gray;
	img = imread("../image/lenna.bmp", IMREAD_COLOR);
	cvtColor(img, gray, COLOR_BGR2GRAY);
	cout << gray.type() << endl;

	imshow("gray", gray);
	waitKey();

	int width = 3;
	int height = 3;
	double sigma = 1.0;
	double *kernel = new double[width * height];
	generateKernel(width, height, sigma, kernel);

	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			cout << kernel[(i * width) + j] << " ";
		}

		cout << endl;
	}

	delete [] kernel;

	return 0;
}
