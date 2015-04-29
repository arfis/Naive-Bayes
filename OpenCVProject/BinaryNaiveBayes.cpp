#include "stdafx.h"
#include "BinaryNaiveBayes.h"
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
#define PI 3.14159265
int BDES_MODE = 4;
void brotate(Mat *picture,int picture_number);
void baddToTrainingSet(Mat descriptors,int number);
vector <KeyPoint> brotateKeyPoints1(vector<KeyPoint> keypoints,int angle,Point center);
vector<vector<std::bitset<256> >> trainedBag;
vector<int> original_descriptor_size;
vector<KeyPoint> *bgetKeypoints(Mat picture);
Mat *bgetDescriptors(Mat picture,vector<KeyPoint> *keyPoints);
float computeValue(std::bitset<256> bites,std::bitset<256> secondBites,int sum_descriptors,int des_class_number);
vector<KeyPoint> *bprocessedKeyPoints;
int sum_descriptors = 0;
int bayes_numberOfClasses;
void BinaryNaiveBayes::init(int size)
{
	bayes_numberOfClasses = size;
	for(int i =0;i<size;i++){
		trainedBag.push_back(vector<std::bitset<256>>());
		original_descriptor_size.push_back(0);
	}
}

BinaryNaiveBayes::BinaryNaiveBayes(void)
{
	
}

BinaryNaiveBayes::~BinaryNaiveBayes(void)
{
}

void BinaryNaiveBayes::trainBayes(Mat *picture,int number){
	
		brotate(picture,number);
		
}
vector<vector<float>> BinaryNaiveBayes::findMatch(Mat picture,int* recognized_class){
	
	float value;
	int rec_class;
	int rec_des;
	vector<int> all_classes(bayes_numberOfClasses,0);

	imshow("obrazok",picture);
	waitKey(0);

	vector<KeyPoint> *keyP = bgetKeypoints(picture);
	Mat *image_descriptors = bgetDescriptors(picture,keyP);

	image_descriptors->convertTo(*image_descriptors, CV_32FC1);
	vector<vector<float>> resultVect(image_descriptors->rows, vector<float>(3));
	

		for(int descriptor_row = 0;descriptor_row<image_descriptors->rows;descriptor_row++){
			String des = "";
			
			for(int descriptor_col = 0;descriptor_col<image_descriptors->cols;descriptor_col++){
				float number = image_descriptors->at<float>(descriptor_row,descriptor_col);
				std::bitset<8> bs (number);
				des += bs.to_string();
			}
			//zoberieme bitset aktualneho deskriptora
			std::bitset<256> bites(des);
			float max = 0.0;

			for(int class_number=0;class_number<trainedBag.size();class_number++){
				for(int descriptor_number=0;descriptor_number<trainedBag[class_number].size();descriptor_number++){
					//porovname tento bitset s kazdym deskriptorom
					value = computeValue(trainedBag[class_number][descriptor_number],bites,sum_descriptors,trainedBag[class_number].size());
					
					if(value>max){
						max			= value;
						rec_class	= class_number;
						rec_des		= descriptor_number;
					}
				}
			}
			resultVect[descriptor_row][0] = rec_class;
			resultVect[descriptor_row][1] = rec_des%(original_descriptor_size[rec_class]);
			resultVect[descriptor_row][2] = value;
			all_classes[rec_class]++;

		}

			int max = 0;
			for(int i =0;i<all_classes.size();i++){
				if(max < all_classes[i]){
					max					= all_classes[i];
					*recognized_class	= i;
				}
			}
		return resultVect;
}

float computeValue(std::bitset<256> bitsetFromBag,std::bitset<256> unknownBitset,int sum_desc,int desc_in_class){
	float distance = (bitsetFromBag^unknownBitset).count();
	float posterior = (float)(sum_desc-desc_in_class)/(float)sum_desc;
	return (float)((float)1/(float)distance)*(float)posterior;
}

void baddToTrainingSet(Mat descriptors,int image_number){
	descriptors.convertTo(descriptors, CV_32FC1);
	for(int descriptor_row = 0;descriptor_row<descriptors.rows;descriptor_row++){
		String des = "";

		for(int descriptor_col = 0;descriptor_col<descriptors.cols;descriptor_col++){
			float number = descriptors.at<float>(descriptor_row,descriptor_col);
			std::bitset<8> bs (number);
			des += bs.to_string();
		}
		sum_descriptors += 1;
		std::bitset<256> bites(des);
		trainedBag[image_number].push_back(bites);
	}

}

void brotate(Mat *picture,int number){
	vector<KeyPoint> *keypoints = bgetKeypoints(*picture);
	vector<KeyPoint> *newKeypoints = new vector<KeyPoint>;
	double angle = -1.0;
	double scale = 1;
	Mat new_picture;
	Mat help;
	Mat mKeypoints;
	Mat rotated_keypoints;

	String original = "original";
	String warped = "warped";
	
	Point center = Point( picture->size().width/2,picture->size().height/2);
	Mat rot_mat = getRotationMatrix2D( center, angle, scale );
	bprocessedKeyPoints = bgetKeypoints(*picture);
	Mat descriptors = *bgetDescriptors(*picture, bprocessedKeyPoints);

	if(original_descriptor_size[number] == 0){
		original_descriptor_size[number] = descriptors.rows;
	}

	baddToTrainingSet(descriptors,number);
	//ShowKeyPoints1(*picture,processedKeyPoints);
	
	for(int i=0;i<40;i++){
		angle = rand() % 360;
		rot_mat = getRotationMatrix2D( center, angle, scale );
		warpAffine( *picture, new_picture, rot_mat, picture->size());
	
		*newKeypoints = brotateKeyPoints1(*bprocessedKeyPoints,angle,center);
		baddToTrainingSet(*bgetDescriptors(new_picture, newKeypoints),number);
	}
	printf("konec");
}

vector <KeyPoint> brotateKeyPoints1(vector<KeyPoint> keypoints,int angle,Point center){
	//RotatedRect myRect;
	std::vector<cv::Point2f> points;
	Point2f onePoint;
	vector<KeyPoint>::iterator it;
	vector<KeyPoint> fKeyPoints;
	float newX,newY;
	float test,test2;
	angle = -angle;
	RotatedRect rect;

	for( it= keypoints.begin(); it!= keypoints.end();it++)
	{
		
		float tempX = it->pt.x - center.x;
		float tempY = it->pt.y - center.y;

		newX = tempX*cos(angle*PI/180.0)-tempY*sin(angle*PI/180.0);
		newY = tempX*sin(angle*PI/180.0)+tempY*cos(angle*PI/180.0);

		newX = newX + center.x;
		newY = newY + center.y;
		onePoint = Point2f(newX,newY);
		//newKeypoints.push_back(cv::KeyPoint(onePoint, 1.f));
		points.push_back(onePoint);
	}

	//konverzia z point2f na keypointy
	for( size_t i = 0; i < points.size(); i++ ) {
		fKeyPoints.push_back(cv::KeyPoint(points[i], 1.f));
	}
	return fKeyPoints;
}
vector<KeyPoint> *bgetKeypoints(Mat picture){

	std::vector<cv::KeyPoint> *keypointsA = new vector<KeyPoint>;
	cv::Ptr<cv::FeatureDetector> detector;

	detector = cv::Algorithm::create<cv::FeatureDetector>("Feature2D.BRISK");
	detector->detect(picture, *keypointsA);
	return keypointsA;
}
Mat *bgetDescriptors(Mat picture,vector<KeyPoint> *keyPoints){
		Mat *descriptors = new Mat;
		cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;

			switch(BDES_MODE){
		case 1:{
			descriptorExtractor =cv::Algorithm::create<cv::DescriptorExtractor>("Feature2D.BRIEF");
			break;
		   }

		case 2:{
			descriptorExtractor =cv::Algorithm::create<cv::DescriptorExtractor>("Feature2D.BRISK");

			break;
		   }

		case 3:{
			descriptorExtractor =cv::Algorithm::create<cv::DescriptorExtractor>("Feature2D.ORB");
			break;
		   }

		case 4:{
			descriptorExtractor =cv::Algorithm::create<cv::DescriptorExtractor>("Feature2D.FREAK");
			break;
		   }
		}
		descriptorExtractor->compute(picture, *keyPoints, *descriptors);
		return descriptors;
}