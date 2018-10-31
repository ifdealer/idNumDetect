
#if !defined LINEF
#define LINEF

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#define PI 3.1415926

class LineFinder {

private:

	cv::Mat img;

	std::vector<cv::Vec4i> lines;

	double deltaRho;
	double deltaTheta;

	int minVote;

	double minLength;

	double maxGap;

public:

	LineFinder() : deltaRho(1), deltaTheta(PI / 180), minVote(10), minLength(0.), maxGap(0.) {}

	//设置寻找步长
	void setAccResolution(double dRho, double dTheta) {

		deltaRho = dRho;
		deltaTheta = dTheta;
	}
	//设置最小投票数
	void setMinVote(int minv) {

		minVote = minv;
	}

	//设置最小长度以及允许的间隙大小
	void setLineLengthAndGap(double length, double gap) {

		minLength = length;
		maxGap = gap;
	}

	std::vector<cv::Vec4i> findLines(cv::Mat& binary) {

		lines.clear();
		//使用霍夫概率变换来找直线
		cv::HoughLinesP(binary, lines, deltaRho, deltaTheta, minVote, minLength, maxGap);

		return lines;
	}

	void drawDetectedLines(cv::Mat &image, cv::Scalar color = cv::Scalar(255, 255, 255)) {

		std::vector<cv::Vec4i>::const_iterator it2 = lines.begin();

		while (it2 != lines.end()) {

			cv::Point pt1((*it2)[0], (*it2)[1]);
			cv::Point pt2((*it2)[2], (*it2)[3]);

			cv::line(image, pt1, pt2, color);

			++it2;
		}
	}

};


#endif
