#pragma once
#include<opencv2\opencv.hpp>
#include<vector>
#include"linefinder.h"
#define PI 3.141592653
using namespace cv;
using namespace std;

/*主要方法是通过寻找身份证中间隔为一定数值的最长直线来判断身份证号码的位置
并通过旋转矩阵的方式解决非水平或角度歪斜的身份证图片无法得到正确位置的情况*/
class FeatureEx {
private:
	Mat src;
	Mat GrayImage;
	Mat BinaryImage;
	Mat contours; //边缘图
	vector<Vec4i> lines;  //存放检测到的直线
	Vec4i LongestLine; //最长直线
	Mat Roi;
	Mat EdgImage;
public:
	FeatureEx() {}
	void SetSrc(Mat& src);
	void FixPic(const Mat& src);
	void ReSizePic(Mat& src, double TargetSize);
	Vec4i Findline(Mat& contours, double length, double gap, int vote);
	void FindEdge();
	Mat FindRoi();
	Mat Feature();
	
};

void FeatureEx::SetSrc(Mat& src) {
	this->src = src;
}

void FeatureEx::ReSizePic(Mat& src, double TargetSize) {
	double w = src.size().width;
	double h = src.size().height;
	if (w > h)
		cv::resize(src, src, cv::Size(TargetSize, TargetSize*h / w), 0, 0, cv::INTER_LINEAR);
	else
		cv::resize(src, src, cv::Size(TargetSize*w / h, TargetSize), 0, 0, cv::INTER_LINEAR);
}


//这里选择使用MESR特征提取来找出边缘，因为如果用Canny方式寻找边缘的话会把身份证整个框都提取出来
//不利于提取最长的直线
void FeatureEx::FindEdge() {
	int height = src.size().height;
	int width = src.size().width;
	EdgImage = Mat::zeros(GrayImage.size(), CV_8U);
	vector<vector<Point> > points;
	vector<Rect> rects;
	Ptr<MSER> ptrMSER = cv::MSER::create(5, 20, 300, 0.25, 0.1);
	ptrMSER->detectRegions(GrayImage, points, rects);
	RNG rng;
	Mat output(GrayImage.size(), CV_8UC3);
	output = Scalar(255, 255, 255);
	for (auto it = points.rbegin(); it != points.rend(); ++it) {
		//使用RNG随机数生成器生成随机颜色
		cv::Vec3b c(rng.uniform(0, 254), rng.uniform(0, 254), rng.uniform(0, 254));
		for (auto itpts = it->begin(); itpts != it->end(); ++itpts) {
			if (output.at<Vec3b>(*itpts)[0] == 255) {
					output.at<Vec3b>(*itpts) = c;
			}
		}
	}
	for (int i = 1; i < height -1; i++)
		for (int j = 1; j < width-1; j++) {
			if (output.ptr<Vec3b>(i)[j] != output.ptr<Vec3b>(i+1)[j]||
				output.ptr<Vec3b>(i)[j] != output.ptr<Vec3b>(i-1)[j]||
				output.ptr<Vec3b>(i)[j] != output.ptr<Vec3b>(i)[j+1]||
				output.ptr<Vec3b>(i)[j] != output.ptr<Vec3b>(i)[j-1])
				this->EdgImage.at<uchar>(i, j) = 254;
		}
	/*namedWindow("1");
	imshow("1", EdgImage);
	waitKey(0);*/
}



void FeatureEx::FixPic(const Mat& src) {
	Mat tmpimg(src);
	//调整为合适处理的尺寸
	ReSizePic(tmpimg,1000.0);
	//转换为单通道灰度图像
	cvtColor(tmpimg, GrayImage, CV_BGR2GRAY);
	////获取边缘图
	//Canny(GrayImage, contours, 125, 350);
	//使用自适应阈值取得二值图
	adaptiveThreshold(GrayImage, BinaryImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 5, 5);
	medianBlur(BinaryImage, BinaryImage, 3);

}

//寻找边缘图中所有满足条件的直线,并且提取出最长的直线
Vec4i FeatureEx::Findline(Mat& contours, double length, double gap,int vote) {
	//调用直线寻找类寻找所有直线
	LineFinder finder;
	finder.setLineLengthAndGap(length, gap);
	finder.setMinVote(vote);
	//从Canny检测到的点集中寻找直线，返回值是4个整数，为两个端点坐标的x,y
	lines = finder.findLines(contours);
	//最长直线的下标
	int Longest = 0;
	int max = 0;

	for (auto it = lines.begin(); it != lines.end(); ++it) {
		Point pt1((*it)[0], (*it)[1]);
		Point pt2((*it)[2], (*it)[3]);
		//计算两点之间的长度，即线段长
		float distance = sqrtf(powf(pt1.x - pt2.x, 2) + powf(pt1.y - pt2.y, 2));
		if (distance > max) {
			max = distance;
			//可以通过迭代器相减获得下标
			Longest = it - lines.begin();
		}
	}
	return lines[Longest];
}


Mat FeatureEx::FindRoi() {
	/*LongestLine = Findline(contours, 100, 30, 70);*/
	//根据这条最长直线在二值图上取得ROI区域
	Mat tmpimg;
	//最长线段两个端点
	Point2f LinePoint1(LongestLine[0]-20, LongestLine[1]);
	Point2f LinePoint2(LongestLine[2]+20, LongestLine[3]);
	Point pt1(LongestLine[0] - 20, LongestLine[1]);
	Point pt2(LongestLine[2] - 20, LongestLine[3]);
	cout << LinePoint1.x << LinePoint2.x << endl;
	
	//组成矩形ROI的四个角的点
	Point2f LeftUp;
	Point2f LeftDown;
	Point2f RightUp;
	Point2f RightDown;
	float a = (float)(LinePoint1.y - LinePoint2.y) / (float)(LinePoint1.x - LinePoint2.x);
	float b = (LinePoint1.y - a*LinePoint1.x);
	float c = (-1.0) / a;
	float d = (LinePoint1.y - c*LinePoint1.x);
	float d_2 = (LinePoint2.y - c*LinePoint2.x);

	float distance = sqrtf(powf(LinePoint1.x - LinePoint2.x, 2) + powf(LinePoint1.y - LinePoint2.y, 2));//线段长
	//线段角度 与水平线的夹角
	float ang;
	ang = (LinePoint1.y - LinePoint2.y) / distance;


	if (LinePoint1.y == LinePoint2.y) { //如果这条线是水平的
		LeftUp.y = pt1.y - 30;
		LeftUp.x = pt1.x;
		LeftDown.y = pt1.y + 30;
		LeftDown.x = pt1.x;
		RightUp.y = pt2.y - 30;
		RightUp.x = pt2.x;

		RightDown.y = pt2.y + 30;
		RightDown.x = pt2.x;
		vector<Point> Area;
		Area.push_back(LeftUp);
		Area.push_back(LeftDown);
		Area.push_back(RightUp);
		Area.push_back(RightDown);;
		RotatedRect rect = minAreaRect(Area);  //根据4个点寻找最小外接旋转矩形
		Point center = rect.center;
		tmpimg = tmpimg(Rect(center.x - (rect.size.height / 2), center.y - (rect.size.width / 2), rect.size.height, rect.size.width));
		/*namedWindow("1");
		imshow("1", tmpimg);
		waitKey(0);*/
		
		return tmpimg;
	}
	else if(ang> (-(2.0/3.0)) && ang < 0 ){ //如果直线角度为(135-180)之间
		LeftUp.y = LinePoint1.y - 30;
		LeftDown.y = LinePoint1.y + 30;

		LeftUp.x = (LeftUp.y - d) / c;
		LeftDown.x = (LeftDown.y - d) / c;

		RightUp.y = LinePoint2.y - 30;
		RightDown.y = LinePoint2.y + 30;

		RightUp.x = (RightUp.y - d_2) / c;
		RightDown.x = (RightDown.y - d_2) / c;
		vector<Point2f> Area;
		Area.push_back(LeftUp);
		Area.push_back(LeftDown);
		Area.push_back(RightUp);
		Area.push_back(RightDown);
		RotatedRect rect = minAreaRect(Area);  //根据4个点寻找最小外接旋转矩形
		//先将图片按照旋转矩形旋转
		Point2f center = rect.center;
		Mat rot_mat = getRotationMatrix2D(center, rect.angle + 90, 1.0);

		Size dst_sz(BinaryImage.size());
		warpAffine(BinaryImage, tmpimg, rot_mat, dst_sz);
		//提取ROI
		tmpimg = tmpimg(Rect(center.x - (rect.size.height / 2), center.y - (rect.size.width / 2), rect.size.height, rect.size.width));
		//消除噪点
		medianBlur(tmpimg, tmpimg, 3);
		return tmpimg;
	}
	else if (ang > 0 && ang < (2.0 / 3.0)) {  //如果直线角度为(0-45)之间
		LeftUp.y = LinePoint1.y - 30;
		LeftDown.y = LinePoint1.y + 30;

		LeftUp.x = (LeftUp.y - d) / c;
		LeftDown.x = (LeftDown.y - d) / c;

		RightUp.y = LinePoint2.y - 30;
		RightDown.y = LinePoint2.y + 30;

		RightUp.x = (RightUp.y - d_2) / c;
		RightDown.x = (RightDown.y - d_2) / c;
		vector<Point2f> Area;
		Area.push_back(LeftUp);
		Area.push_back(LeftDown);
		Area.push_back(RightUp);
		Area.push_back(RightDown);
		RotatedRect rect = minAreaRect(Area);  //根据4个点寻找最小外接旋转矩形
											   //先将图片按照旋转矩形旋转
		Point2f center = rect.center;
		Mat rot_mat = getRotationMatrix2D(center, rect.angle, 1.0);

		Size dst_sz(BinaryImage.size());
		warpAffine(BinaryImage, tmpimg, rot_mat, dst_sz);
		//提取ROI
		tmpimg = tmpimg(Rect(center.x - (rect.size.width / 2), center.y - (rect.size.height / 2), rect.size.width, rect.size.height));
		//消除噪点
		medianBlur(tmpimg, tmpimg, 3);
		return tmpimg;
		
	}
}

Mat FeatureEx::Feature() {
	ReSizePic(src, 1000.0);
	FixPic(src);
	FindEdge();
	LongestLine = Findline(EdgImage, 100, 30, 70);
	Roi = FindRoi();
	return Roi;
}
