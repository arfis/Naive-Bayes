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
#include <bitset>

using namespace cv;

class BinaryNaiveBayes
{
public:
	BinaryNaiveBayes(void);
	vector<vector<float>> findMatch(Mat picture,int *recognizedPoint);
	void trainBayes(Mat *picture,int number);
	~BinaryNaiveBayes(void);
	void init(int size);
};


