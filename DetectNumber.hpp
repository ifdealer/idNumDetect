#pragma once
#include<opencv2\opencv.hpp>
#include<sstream>
#include<string>
#include<fstream>

using namespace cv;
using namespace std;
using namespace cv::ml;

class Detect {
private:
	Mat src;
	Mat data;  // 特征矩阵
	Mat labels;
	int K;
	Size DataSize;
	vector<int> idnumber;
	Ptr<TrainData> tData;
	Ptr<KNearest> model;
	vector<Rect> rect;  //存放每个数字的矩形

public:
	Detect() :K(5), DataSize(Size(45, 45)) {}
	Detect(Mat& src) :K(5), DataSize(Size(45, 45)),src(src) {}
	//存放检测到的数字
	vector<int> DetectNum();  
	//KNN训练模型
	void KNNTrainData();

	void SetK(int k);
	void SetSrc(Mat& src);

	vector<Rect> FindRect(const Mat& src); //找到每个数字的矩形区域
	void ShowNum();

	//比较函数
	static bool comp_rect(const Rect& rect1, const Rect& rect2) {
		return rect1.tl().x < rect2.tl().x;
	}
	
};

void Detect::SetK(int k) {
	this->K = k;
}

void Detect::SetSrc(Mat& src) {
	this->src = src;
}

void Detect::KNNTrainData() {
	string numfilename;
	int flag = 0;
	for (flag = 0; flag < 10; flag++) {
		string filename;
		stringstream sstr;
		sstr << flag;
		filename = sstr.str();
		numfilename = "H:\\数字训练集\\num" + filename + "\\Eng.txt";
		string ImgName;
		ifstream fin(numfilename);//打开原始样本图片文件列表  
		while (getline(fin, ImgName)) {
			Mat getnum;
			/*	cout << "processing:" << ImgName << endl;*/
			ImgName = "H:\\数字训练集\\num" + filename + "\\" + ImgName;
			getnum = imread(ImgName);

			cvtColor(getnum, getnum, CV_BGR2GRAY);
			threshold(getnum, getnum, 128, 255, CV_THRESH_BINARY);

			resize(getnum, getnum, DataSize, 0, 0, INTER_LINEAR);
			data.push_back(getnum.reshape(0, 1));
			labels.push_back(flag);
		}
	}
	data.convertTo(data, CV_32FC1);  //import
	tData = TrainData::create(data, ROW_SAMPLE, labels);
	model = KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(tData);
}

vector<Rect> Detect::FindRect(const Mat& src) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarhy;
	findContours(src, contours, hierarhy, RETR_LIST, CHAIN_APPROX_NONE);
	// 删除太短或太长的轮廓
	int cmin = 50; // 最小轮廓长度
	int cmax = 400; // 最大轮廓长度
	auto itc = contours.begin();

	// 针对所有轮廓
	while (itc != contours.end()) {
		// 验证轮廓大小
		if (itc->size() < cmin || itc->size() > cmax)
			itc = contours.erase(itc);
		else {
			Rect r1 = boundingRect(*itc);
			//将矩形稍微扩大
			r1 = r1 + Point(-2, -2);
			r1 = r1 + Size(4, 4);
			/*rectangle(result_1, r1, 0, 1);*/
			rect.push_back(r1);
			++itc;
		}
	}
	//对图片中的数字按X坐标进行排序
	sort(this->rect.begin(),this->rect.end(),Detect::comp_rect);
	return rect;

}


vector<int> Detect::DetectNum() {
	KNNTrainData();
	FindRect(src);
	for (auto it = rect.begin(); it != rect.end(); ++it) {
		Mat roi = src(*it); //获取单个数字
		roi.convertTo(roi, CV_32FC1);

		threshold(roi, roi, 1, 255, CV_THRESH_BINARY);
		int x = (DataSize.width - roi.cols) / 2;
		int y = (DataSize.height - roi.rows) / 2;
		Mat sbackground(DataSize, CV_32FC1, Scalar(0));
		Mat newroi(sbackground, Rect(x, y, roi.cols, roi.rows));
		roi.copyTo(newroi);
		sbackground = sbackground.reshape(0, 1);
		float r = model->predict(sbackground);
		idnumber.push_back((int)r);
	}
	return idnumber;
}

void Detect::ShowNum() {
	for (auto it : idnumber) {
		cout << it;
	}
}