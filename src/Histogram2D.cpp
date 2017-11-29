#include"Histogram2D.h"

Histogram2D::Histogram2D() {
	hist = new double*[hbins];
	for (int i = 0; i < hbins; ++i) {
		hist[i] = new double[sbins];
		double* p_hist = hist[i];
		for (int j = 0; j < sbins; ++j) {
			p_hist[j] = 0.0;
		}
	}
}

Histogram2D::~Histogram2D() {
	for (int i = 0; i < hbins; ++i) {
		delete[] hist[i];
	}
	delete hist;
}

void Histogram2D::init() {
	for (int i = 0; i < hbins; ++i) {
		double* p_hist = hist[i];
		for (int j = 0; j < sbins; ++j) {
			p_hist[j] = 0.0;
		}
	}
}

void Histogram2D::calcHist(cv::Mat& hsv, cv::Rect& roi) {
	init();

	for (int y = roi.y, Y = roi.y + roi.height; y < Y; ++y) {
		cv::Vec3b* p_hsv = hsv.ptr<cv::Vec3b>(y);
		for (int x = roi.x, X = roi.x + roi.width; x < X; ++x) {
			int h = (int)p_hsv[x][0] / hsize;
			int s = (int)p_hsv[x][1] / ssize;
			hist[h][s]++;
		}
	}

	norm();
}

void Histogram2D::calcWeightedHist(cv::Mat& hsv, cv::Rect& roi, cv::Mat& kernel) {
	init();

	int startX = roi.x, startY = roi.y;

	for (int y = 0, Y = roi.height; y < Y; ++y) {
		cv::Vec3b* p_hsv = hsv.ptr<cv::Vec3b>(startY + y);
		double* p_kernel = kernel.ptr<double>(y);
		for (int x = 0, X = roi.width; x < X; ++x) {
			int h = (int)p_hsv[startX + x][0] / hsize;
			int s = (int)p_hsv[startX + x][1] / ssize;
			hist[h][s] += p_kernel[x];
		}
	}

	norm();
}

void Histogram2D::showHist() {
	for (int i = 0; i < hbins; ++i) {
		double* p_hist = hist[i];
		for (int j = 0; j < sbins; ++j) {
			printf("%2.1e ", p_hist[j]);
		}
		printf("\n");
	}
}

void Histogram2D::norm() {
	double sum = 0.0;
	for (int i = 0; i < hbins; ++i) {
		double* p_hist = hist[i];
		for (int j = 0; j < sbins; ++j) {
			sum += p_hist[j];
		}
	}

	for (int i = 0; i < hbins; ++i) {
		double* p_hist = hist[i];
		for (int j = 0; j < sbins; ++j) {
			p_hist[j] /= sum;
		}
	}
}
