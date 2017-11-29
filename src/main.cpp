/*
	Mean Shift Tracking
		writen by T. Itazuri

	Description
		ÅEHistogram is 64[bin]x64[bin] array and uses Hue and Saturation in HSV color space.
		ÅEKernel function is normalized Gaussian kernel.
*/

#include<iostream>
#include<opencv2\opencv.hpp>
#include<limits>

#include"Histogram2D.h"
#include"GaussianKernel.h"

cv::Mat img, img_roi, hsv;
std::string select_winname = "Target Object Selection", target_winname = "Target Object", search_winname = "Searching...";
cv::Point2i sp(-1, -1), ep(-1, -1);
bool bSelecting = false, bSelected = false;

cv::Mat kernel;
Histogram2D target_hist, candidate_hist;

cv::Rect pre_roi, cur_roi;
double pre_rho, cur_rho;
const double eps = 1.0;

void onMouse(int event, int x, int y, int flags, void* userdata);
double calcBhattacharyyaCoeff(Histogram2D& _h1, Histogram2D& _h2);
cv::Vec2d calcMeanShiftVector(cv::Mat& _hsv, Histogram2D& _target_hist, Histogram2D& _candidate_hist, cv::Rect& _roi, cv::Mat& _kernel);
double calcDist(cv::Rect& _r1, cv::Rect& _r2);

int main(int argc, char* argv[]) {
	// Load Video
	std::string str = "occulusion_30fps";
	cv::VideoCapture video(str + ".mp4");
	cv::VideoWriter writer(str + "_tracking.avi", CV_FOURCC('X', 'V', 'I', 'D'), video.get(CV_CAP_PROP_FPS), cv::Size(video.get(CV_CAP_PROP_FRAME_WIDTH), video.get(CV_CAP_PROP_FRAME_HEIGHT)));

	// Target Object Selection
	video >> img;
	cv::imshow(select_winname, img);
	cv::setMouseCallback(select_winname, onMouse);
	cv::waitKey();
	cv::destroyWindow(select_winname);

	// Create Gaussian Kernel and Calculate Target Histogram
	cv::Rect target_roi(sp, ep);
	kernel = createGaussianKernel(target_roi);
	cv::cvtColor(img, hsv, CV_BGR2HSV_FULL);
	target_hist.calcWeightedHist(hsv, target_roi, kernel);

	// Target Object
	cv::imshow(target_winname, (img.clone())(target_roi));

	// Initialization
	pre_roi = cv::Rect(sp, ep);

	while (1) {
		video >> img;
		char key = cv::waitKey(1);
		if (img.empty() || key == 27) break;

		cv::cvtColor(img, hsv, CV_BGR2HSV_FULL);

		while (1) {
			// Calculate Histogram Before Update
			candidate_hist.calcWeightedHist(hsv, pre_roi, kernel);

			// Calculate Bhattacharyya Coefficient Before Update
			pre_rho = calcBhattacharyyaCoeff(target_hist, candidate_hist);

			// Calculate Mean Shift Vector
			cv::Vec2d msv = calcMeanShiftVector(hsv, target_hist, candidate_hist, pre_roi, kernel);

			// Calculate Candidate Position
			cur_roi = cv::Rect(pre_roi.x + (int)(msv[0]), pre_roi.y + (int)(msv[1]), pre_roi.width, pre_roi.height);
			if (cur_roi.x < 0 || cur_roi.y < 0 || cur_roi.x + cur_roi.width > img.cols || cur_roi.y + cur_roi.height > img.rows) {
				cur_roi = pre_roi;
				break;
			}

			// Calculate Histogram located at Candidate Position
			candidate_hist.calcWeightedHist(hsv, cur_roi, kernel);

			// Calculate Bhattacharyya Coefficient located at Candidate Position
			cur_rho = calcBhattacharyyaCoeff(target_hist, candidate_hist);

			//img_roi = img.clone();
			//cv::rectangle(img_roi, pre_roi, cv::Scalar(255, 0, 0), 2, 8, 0);
			//cv::rectangle(img_roi, cur_roi, cv::Scalar(0, 0, 255), 2, 8, 0);
			//cv::imshow(search_winname, img_roi);
			//cv::waitKey();

			while (cur_rho + 1e-2 < pre_rho) {
				// Calculate Candidate Position
				cur_roi = cv::Rect((cur_roi.x + pre_roi.x) / 2, (cur_roi.y + pre_roi.y) / 2, cur_roi.width, cur_roi.height);

				// Calculate Histogram located at Candidate Position
				candidate_hist.calcWeightedHist(hsv, cur_roi, kernel);

				// Calculate Bhattacharyya Coefficient located at Candidate Position
				cur_rho = calcBhattacharyyaCoeff(target_hist, candidate_hist);

				//img_roi = img.clone();
				//cv::rectangle(img_roi, pre_roi, cv::Scalar(255, 0, 0), 2, 8, 0);
				//cv::rectangle(img_roi, cur_roi, cv::Scalar(0, 0, 255), 2, 8, 0);
				//cv::imshow(search_winname, img_roi);
				//cv::waitKey();
			}

			// Evaluation
			if (calcDist(pre_roi, cur_roi) < eps) break;
			else pre_roi = cur_roi;
		}

		//std::cout << "Calculated!" << std::endl;
		cv::rectangle(img, cur_roi, cv::Scalar(0, 0, 255), 2, 8, 0);
		cv::imshow("img", img);
		writer << img;

		pre_roi = cur_roi;
	}

	return 0;
}

void onMouse(int event, int x, int y, int flags, void* userdata) {
	img_roi = img.clone();

	switch (event)
	{
	case cv::EVENT_LBUTTONDOWN:
		sp = cv::Point2i(x, y);
		bSelecting = true;
		bSelected = false;
		break;

	case cv::EVENT_LBUTTONUP:
		ep = cv::Point2i(x, y);
		cv::rectangle(img_roi, sp, ep, cv::Scalar(255, 0, 0), 2, 8, 0);
		bSelecting = false;
		bSelected = true;
		break;

	case cv::EVENT_MOUSEMOVE:
		if (bSelecting) {
			ep = cv::Point2i(x, y);
			cv::rectangle(img_roi, sp, ep, cv::Scalar(255, 0, 0), 2, 8, 0);
		}

		if (bSelected) {
			cv::rectangle(img_roi, sp, ep, cv::Scalar(0, 0, 255), 2, 8);
		}
		break;
	}

	cv::imshow(select_winname, img_roi);
}

double calcBhattacharyyaCoeff(Histogram2D& h1, Histogram2D& h2) {
	double rho = 0.0;

	for (int i = 0; i < hbins; ++i) {
		double* p_h1 = h1.hist[i];
		double* p_h2 = h2.hist[i];
		for (int j = 0; j < sbins; ++j) {
			rho += std::sqrt(p_h1[j] * p_h2[j]);
		}
	}

	return rho;
}

cv::Vec2d calcMeanShiftVector(cv::Mat& _hsv, Histogram2D& _target_hist, Histogram2D& _candidate_hist, cv::Rect& _roi, cv::Mat& _kernel) {
	cv::Vec2d msv(0, 0);

	int startX = _roi.x, startY = _roi.y;
	int centerX = (_roi.width - 1) / 2.0, centerY = (_roi.height - 1) / 2.0;

	double sum = 0.0;
	for (int y = 0, Y = _roi.height; y < Y; ++y) {
		cv::Vec3b* _p_hsv = _hsv.ptr<cv::Vec3b>(y + startY);
		double* _p_kernel = _kernel.ptr<double>(y);

		for (int x = 0, X = _roi.width; x < X; ++x) {
			cv::Vec3b val = _p_hsv[startX + x];
			int hbin = val[0] / hsize;
			int sbin = val[1] / ssize;
			if (_candidate_hist.hist[hbin][sbin] > std::numeric_limits<double>::epsilon()) {
				double weight = _p_kernel[x] * std::sqrt(_target_hist.hist[hbin][sbin] / _candidate_hist.hist[hbin][sbin]);
				sum += weight;
				msv += weight * cv::Vec2d(x - centerX, y - centerY);
			}
		}
	}
	msv /= sum;
	//std::cout << "Mean Shift Vector: " << msv << std::endl;

	return msv;
}

double calcDist(cv::Rect& r1, cv::Rect& r2) {
	return std::sqrt((r1.x - r2.x) * (r1.x - r2.x) + (r1.y - r2.y) * (r1.y - r2.y));
}