#pragma once
#include<iostream>
#include<opencv2\opencv.hpp>

cv::Mat createGaussianKernel(cv::Rect& win) {
	int X = win.width;
	int Y = win.height;

	double centerX = (X - 1) / 2.0;
	double centerY = (Y - 1) / 2.0;

	cv::Mat kernel(cv::Size(X, Y), CV_64FC1);

	double sum = 0.0;
	for (int y = 0; y < Y; ++y) {
		double* p_kernel = kernel.ptr<double>(y);
		for (int x = 0; x < X; ++x) {
			double buf = std::exp(-0.5 * (std::pow((x - centerX) / centerX, 2.0) + std::pow((y - centerY) / centerY, 2.0)));
			p_kernel[x] = buf;
			sum += buf;
		}
	}

	// initialization
	for (int y = 0; y < Y; ++y) {
		double* p_kernel = kernel.ptr<double>(y);
		for (int x = 0; x < X; ++x) {
			p_kernel[x] /= sum;
		}
	}

	return kernel;
}

