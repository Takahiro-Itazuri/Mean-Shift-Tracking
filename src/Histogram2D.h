#pragma once
#include<iostream>
#include<opencv2\opencv.hpp>

const int hbins = 64;
const int sbins = 64;
const int hsize = 256 / hbins;
const int ssize = 256 / sbins;

class Histogram2D {
public:
	Histogram2D();
	~Histogram2D();
	void init();
	void calcHist(cv::Mat& hsv, cv::Rect& roi);
	void calcWeightedHist(cv::Mat& hsv, cv::Rect& roi, cv::Mat& kernel);
	void showHist();

	double** hist;
private:
	void norm();
};
