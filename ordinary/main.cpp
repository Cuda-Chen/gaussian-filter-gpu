#include <iostream>
#include <ctime>

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
	//cout << gray.type() << endl;

	//imshow("gray", gray);
	//waitKey();

	int width = gray.cols;
	int height = gray.rows;
	unsigned char *source = new unsigned char[height * width];
	unsigned char *destination = new unsigned char[height * width];
	int kernelWidth = 9;
	int kernelHeight = 9;
	double sigma = 3.0;

	//memset(source, 0x0, height * width);
	//memset(destination, 0x0, height * width);
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			source[(i * width) + j] = gray.at<uchar>(i, j);
			destination[(i * width) + j] = source[(i * width) + j];
		}
	}

	//cout << kernelWidth / 2 << endl;
	clock_t begin = clock();
	gaussianFilter(source, destination, width, height, kernelWidth, kernelHeight, sigma);
	clock_t end = clock();
	cout << "spend " << double(end - begin) / CLOCKS_PER_SEC << "seconds" << endl;

	Mat result(height, width, CV_8UC1, destination);
	/*for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			result.at<unsigned char>(i, j) = destination[(i * width) + j];
		}
	}*/

	//Mat opencvBlur;
	//GaussianBlur(gray, opencvBlur, Size(kernelWidth, kernelHeight), sigma, 0);

	imshow("gaussian", result);
	//imshow("opencv gaussian", opencvBlur);
	waitKey();

	delete [] source;
	delete [] destination;

	return 0;
}
