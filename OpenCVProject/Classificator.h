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
#include "Fern.h"

using namespace cv;
class Classificator
{
public:
	Classificator(void);
	~Classificator(void);
	void train(Mat trainingData,Mat trainingClasses,int number);
	vector<int> identify(Fern *f,int mode);	
	vector<int> compute(Fern *trained);

	int predict(Mat data);
};

