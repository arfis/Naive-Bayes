#pragma once
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

class Fern
{
	float *average;
	float *variance;
	int Ksize;
	int fernSize;
public:
	vector< vector <int> > dataHistogram;
	vector<int> trainedHistogram;
	//float *average;
	//float *variance;
	Fern::Fern(int numberOfClasses);
	void train(Mat* picture,int number);
	int numberOfClasses;
	void loadFromFile();
	int size;
	void loadUnknown(Mat data);
	Fern(Mat data, Mat classes,int count);
	Fern(void);
	Fern(Mat data);
	void delet();
	void printVariance();
	~Fern(void);
	String getName(void);
	void showHistogram(void);
	vector<vector<float>> recognize(Mat descriptors,int *point,int write,float *probability);
	void setTest(int i);
	long binary_decimal(string num);
	//vector<vector<int>> Fern(Mat data, Mat classes,int count);
};

