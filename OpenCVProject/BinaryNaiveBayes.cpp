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
int MIN_PROBABILITY = 80;
int BROTATION_COUNT = 40;
void brotate(Mat *picture,int picture_number);
void baddToTrainingSet(Mat descriptors,int number);
vector <KeyPoint> brotateKeyPoints1(vector<KeyPoint> keypoints,int angle,Point center);
vector<vector<std::bitset<512> >> trainedBag;
vector<int> original_descriptor_size;
vector<KeyPoint> *bgetKeypoints(Mat picture);
Mat *bgetDescriptors(Mat picture,vector<KeyPoint> *keyPoints);
float computeValue(std::bitset<512> bites,std::bitset<512> secondBites);
float computePosterior(int sum_desc,int desc_in_class);
vector<KeyPoint> *bprocessedKeyPoints;
int sum_descriptors = 0;
int bayes_numberOfClasses;

void BinaryNaiveBayes::init(int size)
{
	bayes_numberOfClasses = size;
	for(int i =0;i<size;i++){
		trainedBag.push_back(vector<std::bitset<512>>());
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
vector<vector<float>> BinaryNaiveBayes::findMatch(Mat picture,int* recognized_class,float* probability){
	
	float value;
	int rec_class;
	int rec_des;
	vector<int> all_classes(bayes_numberOfClasses,0);
	vector<vector<float>> sameValues;
	
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
			std::bitset<512> bites(des);
			float max = 0.0;
			int same_index = 0;

			for(int class_number=0;class_number<trainedBag.size();class_number++){
				for(int descriptor_number=0;descriptor_number<trainedBag[class_number].size();descriptor_number++){
					//porovname tento bitset s kazdym deskriptorom
					value = computeValue(trainedBag[class_number][descriptor_number],bites);
					//value = 1-value;
					value = value * 100;
				
					if(value>max){
						same_index = 0;
						sameValues.clear();
						max			= value;
						//std::cout << "trieda: " << class_number << value << std::endl;
						sameValues.push_back(vector<float>(3));
						sameValues[same_index][0] = class_number;
						sameValues[same_index][1] = descriptor_number;
						sameValues[same_index][2]= value;
						rec_class	= class_number;
						rec_des		= descriptor_number;
					}

					//pridavanie do zoznamu pointov ktore maju rovnaku vzdialenost
					else if(value == max){
						same_index++;
						sameValues.push_back(vector<float>(3));
						sameValues[same_index][0] = class_number;
						sameValues[same_index][1] = descriptor_number;
						sameValues[same_index][2]= value;	
					}
				}
			}

			if(sameValues.size()>1){
				int maxValue = 0;
				int posterior;
				for(int i=0;i<sameValues.size();i++){
					posterior = computePosterior(sum_descriptors,trainedBag[sameValues[i][0]].size());
					if(maxValue < posterior){
						maxValue = posterior;
						rec_class = sameValues[i][0];
						rec_des = sameValues[i][1];
						max = sameValues[i][2];
					}
				}
			}

			if(max > MIN_PROBABILITY){
				all_classes[rec_class]++;
				resultVect[descriptor_row][0] = rec_class;
				resultVect[descriptor_row][1] = rec_des%(original_descriptor_size[rec_class]);
				resultVect[descriptor_row][2] = max;
			}

			else{
			resultVect[descriptor_row][0] = -1;
			resultVect[descriptor_row][1] = -1;
			resultVect[descriptor_row][2] = -1;
			
			}

		}

			float max = 0;
			*recognized_class = -1;
			int number_of_rec = 0;

			for(int i=0;i<all_classes.size();i++){
				number_of_rec += all_classes[i];
			}
			vector<float> helpVector(bayes_numberOfClasses,0);

			for(int j =0;j<resultVect.size();j++){
				if(resultVect[j][0] != -1)
				helpVector[resultVect[j][0]] += resultVect[j][2];
			}

			for(int j =0;j<all_classes.size();j++){
				if(helpVector[j] != -1){
					helpVector[j] = helpVector[j]/all_classes[j];
				}
			}

			for(int i =0;i<helpVector.size();i++){
				if(max < helpVector[i] && all_classes[i] > 4){
					max					= helpVector[i];
					*recognized_class	= i;
					*probability = max;
				}
			}

		return resultVect;
}

float computeValue(std::bitset<512> bitsetFromBag,std::bitset<512> unknownBitset){
	float distance = (bitsetFromBag^unknownBitset).count();
	//return (float)((float)1/(float)distance)*(float)posterior;
	//float result = (float)((float)((1-((float)distance/(float)512)) * (1-(float)posterior)));
	return (1-((float)distance/(float)512));
	// result;
}
float computePosterior(int sum_desc,int desc_in_class){
	float posterior = (float)(sum_desc-desc_in_class)/(float)sum_desc;
	return posterior;
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
		std::bitset<512> bites(des);
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
	
	for(int i=0;i<ROTATION_COUNT;i++){
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