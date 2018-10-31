#include<opencv2\opencv.hpp>
#include"FeatureExtraction.hpp"
#include"DetectNumber.hpp"
using namespace cv;
int main() {
	Mat original = imread("idcard.png");
	FeatureEx feature;
	feature.SetSrc(original);
	Mat result = feature.Feature();

	Detect detect;
	detect.SetSrc(result);
	detect.DetectNum();
	detect.ShowNum();
	namedWindow("1");
	imshow("1", result);
	waitKey(0);
	

	return 0;
}