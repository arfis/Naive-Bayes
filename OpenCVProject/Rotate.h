#pragma once
#include "Fern.h"

class Rotate
{
public:
	Rotate(void);
	~Rotate(void);
	Mat rotateImage(const Mat &fromI, Mat *toI, const Rect &fromroi, double angle);
	vector <KeyPoint> rotateKeyP(vector<KeyPoint> keypoints,int angle,Point center);
};

