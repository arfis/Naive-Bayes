#include "stdafx.h"
#include "Fern.h"
#include <math.h>
#include "Classificator.h"
#include <stdlib.h>
#include <stdio.h>
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
#include <iostream> // library that contain basic input/output functions
#include <fstream> 

//TO DO: -add to the histogram a new column which tells that from which picture it is
//-adding all keypoints to one big two dimensional array
using namespace std;

#define PI 3.14159265
char* TRAIN_FILE = "training_semiNaive.txt";
char* RANDOM_FILE = "random_numbers.txt";
int const ROTATION_COUNT = 0;
int const FERN_COUNT = 40;
int FERN_SIZE = 10;
int DES_MODE = 4;
int no;
//int Ksize;
long binary_to_decimal(string num);
//vector<int> normalize(vector<int> vect,int cols);
void showPicture(Mat picture,char *name);
void writeToFile(int mode);
vector< vector <int> > *dataHistogram = new vector<vector<int>>();
vector<int> *trainedHistogram = new vector<int>();
void ShowKeyPoints1(Mat picture,vector<KeyPoint> *keypointsA);
void addToTrainingSet(Mat descriptors,int numberClass);
vector <KeyPoint> rotateKeyPoints1(vector<KeyPoint> keypoints,int angle,Point center);

vector<vector<vector<int>>> histogram;
vector<char> wasSet;
vector<KeyPoint> *processedKeyPoints;
vector<int> randomNumbers(vector<int>(40));
int classes;
int wasCreated = 0;
int test = 0;
int numberOfClasses;
cv::String name;

Fern::Fern()
{
}


void Fern::loadFromFile(){
	string line;
	int type = 1;
	int readingClass = -1;
	int readingDescriptor = -1;
	int parsedNumber;
	int classOrDes;
	string delimiter = ";";
	string token;

  ifstream myfile (TRAIN_FILE);
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
		int unknownNumb = stoi(line);

		if(line.find_first_of(";") == -1){
			if(readingClass == -1){
				type++;
				readingClass = unknownNumb;
			}
			//nacitanie jednotlivych deskriptorov - cisla deskriptorov vzdy nacitane cislo musi byt vacsie ako aktualne cislo des
			else if(readingDescriptor < unknownNumb){
				readingDescriptor = unknownNumb;
				histogram[readingClass].push_back( std::vector<int>() );
				histogram[readingClass][readingDescriptor] = vector<int>(FERN_COUNT*pow(2,FERN_SIZE));
			}
			else{
				readingDescriptor = -1;
				readingClass = unknownNumb;
			}

		}
		else{
		
		size_t pos = 0;
		int hist_position = 0;

		while ((pos = line.find(delimiter)) != std::string::npos) {
			token = line.substr(0, pos);
			line.erase(0, pos + delimiter.length());
			if(token.size()>0)
				histogram[readingClass][readingDescriptor][hist_position] = stoi(token);
			hist_position++;
			}
		}
    }
    myfile.close();
  }

  else cout << "Unable to open file"; 
}
void Fern::loadUnknown(Mat data){
	String des;
	String fewBites;
	std::vector<int> histogram(pow(2,FERN_SIZE));

	Mat result;
	long decimal;

	for(int i =0;i<data.rows;i++){
		for(int j=0;j<data.cols;j++){

			float number = data.at<float>(i,j);
			
			std::bitset<9> bs (number);
			des = bs.to_string();

			for(int binaryIndex=0;binaryIndex<FERN_SIZE;binaryIndex++){
				fewBites = des.substr(binaryIndex,FERN_SIZE);
				decimal = binary_to_decimal(fewBites);
				binaryIndex = binaryIndex + FERN_SIZE-1;
				histogram[(int)decimal]++;
			}
	  }
	}
	//trainedHistogram = normalize(histogram,(int)pow(2,8));
}

vector<KeyPoint> *getKeypoints(Mat picture){

	std::vector<cv::KeyPoint> *keypointsA = new vector<KeyPoint>;
	cv::Ptr<cv::FeatureDetector> detector;

	detector = cv::Algorithm::create<cv::FeatureDetector>("Feature2D.BRISK");
	detector->detect(picture, *keypointsA);
	return keypointsA;
}

Mat *getDescriptors(Mat picture,vector<KeyPoint> *keyPoints){
		Mat *descriptors = new Mat;
		cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;

			switch(DES_MODE){
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
		//TODO: Q:preco sa meni pocet keypointov v tejto metode?
		descriptorExtractor->compute(picture, *keyPoints, *descriptors);
		return descriptors;
}

void rotate(Mat *picture,int number){
	vector<KeyPoint> *keypoints = getKeypoints(*picture);
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
	processedKeyPoints = getKeypoints(*picture);
	Mat descriptors = *getDescriptors(*picture, processedKeyPoints);
	int scaleWidth, scaleHeight;

	addToTrainingSet(descriptors,number);
	//ShowKeyPoints1(*picture,processedKeyPoints);
	
	for(int i=0;i<ROTATION_COUNT;i++){
		angle = rand() % 360;

		scaleWidth = rand() % picture->cols;
		scaleHeight = rand() % picture->rows;
		Size size(scaleWidth,scaleHeight);

		rot_mat = getRotationMatrix2D( center, angle, scale );

		warpAffine( *picture, new_picture, rot_mat, picture->size());
	
		*newKeypoints = rotateKeyPoints1(*processedKeyPoints,angle,center);
		addToTrainingSet(*getDescriptors(new_picture, newKeypoints),number);
	}
	printf("konec");
}

void writeToFile(int mode){
	if(mode == 1){
		ofstream fout(TRAIN_FILE);
		if(fout.is_open())
		{
			//file opened successfully so we are here
			cout << "File Opened successfully!!!. Writing data from array to file" << endl;
			int i=0;
			int j=0;
			int f=0;
			int size = pow(2,FERN_SIZE);
			for(int i=0;i<histogram.size();i++){
				fout << i << endl;
				for(int deskriptor=0;deskriptor<histogram[i].size();deskriptor++){
					fout << deskriptor << endl;
					for(int fern = 0;fern<histogram[i][deskriptor].size();fern++){
						 fout << histogram[i][deskriptor][fern]; //writing ith character of array in the file
						 fout << ";";
					}
					fout << endl;
				}
			}	
			cout << "Array data successfully saved into the file" << endl;
		}
		else //file could not be opened
		{
			cout << "File could not be opened." << endl;
		}
	}
	//ulozenie nahodnych cisel do suboru
	if(mode == 2){
		ofstream fout(RANDOM_FILE);
		
		if(fout.is_open())
		{
			//file opened successfully so we are here
					for(int number = 0;number<randomNumbers.size();number++){
						 fout << randomNumbers[number] << endl;//writing ith character of array in the file
					}
			cout << "Array data successfully saved into the file" << endl;		
		}	
		else //file could not be opened
		{
			cout << "File could not be opened." << endl;
		}
	}
}
void matchTwoImages(Mat descriptors_1,Mat descriptors_2){
	
}
Fern::Fern(int numberOfClasses){
	histogram = vector<vector<vector<int>>> (numberOfClasses);
	wasSet = vector<char>(numberOfClasses);
	classes = numberOfClasses;
	//nadstavenie ze ziadna clasa nemala vytvoreny druhy rozmer
	for(int i =0;i<numberOfClasses;i++){
		wasSet.push_back(0);
	}
	 }

void initRandomArray(){
	string line;

	int randomChosingNumber;
	int index	= 0;
	int binSum	= pow(2,9);
	
  ifstream myfile (RANDOM_FILE);

	if (myfile.is_open())
	{	
		while ( getline (myfile,line) )
		{
			int number = stoi(line);
			randomNumbers[index] = number;
			index++;
		}
		myfile.close();
		cout << "nahodne cisla boli uspesne nacitane zo suboru";
	}
	
	if(index == 0){

		for(int binaryIndex=0;binaryIndex<40;binaryIndex++){
			
				randomChosingNumber = rand() % binSum;
				randomNumbers[binaryIndex] = randomChosingNumber;

				if(randomNumbers[binaryIndex] + FERN_SIZE > binSum ){
					randomNumbers[binaryIndex] -= FERN_SIZE;
				}

				else if(randomNumbers[binaryIndex] - FERN_SIZE < 0){
					randomNumbers[binaryIndex] += FERN_SIZE;
				}
		}
		writeToFile(2);
		myfile.close();
		cout<< "nahodne cisla boli uspesne zapisane do suboru";
	}
}

void addToTrainingSet(Mat descriptors,int classNumber){

	descriptors.convertTo(descriptors, CV_32FC1);
	//treba v konstruktore inicializovat trojrozmerne pole podla poctu nacitanych tried obrazkov
	//vector<vector<vector<int>>> histogram(numberClass,vector<vector<int>>(descriptors.rows,vector <int>(40*pow(2,8))));
	Mat result;
	long decimal;
	String des;
	String fewBites;

	int size = pow(2,FERN_SIZE);

	//inicializacia 40 nahodnych cisel - pozicia fernov
	if(wasCreated == 0){
		wasCreated = 1;
		initRandomArray();
	}

	//ak uz boli vytvorene pre danu classu riadky pre deskriptory tak pre zrotovane body sa pouziva uz vytvorene
	if(wasSet[classNumber] == 0){
		wasSet[classNumber] = 1;
		for(int point =0;point<descriptors.rows+1;point++){
			histogram[classNumber].push_back( std::vector<int>() );
			histogram[classNumber][point] = vector<int>(40*pow(2,FERN_SIZE));
		}
	}

	//***************Trenovanie do histogramu*********************//
	for(int point =0;point<descriptors.rows;point++){
		des = "";
		//vytvorenie binarneho retazca z deskriptoru
		for(int j=0;j<descriptors.cols;j++){

			float number = descriptors.at<float>(point,j);
			std::bitset<9> bs (number);
			des += bs.to_string();
	  }
		//pouzitie 40tich nahodnych fernov pri budovani histogramu - 
		//hodnota kazdeho fernu je posunuta o max velkost fernu
		
		for(int fernIndex=0; fernIndex<40; fernIndex++){						
				fewBites = des.substr(randomNumbers[fernIndex],FERN_SIZE);
				decimal = binary_to_decimal(fewBites);
				int position = decimal+(size*fernIndex);
				histogram[classNumber][point][position]++;
			}
	}
}
void ShowKeyPoints1(Mat picture,vector<KeyPoint> *keypointsA){
	Mat outpt;
	cv::drawKeypoints(picture, *keypointsA, outpt, Scalar::all(10), DrawMatchesFlags::DEFAULT);
	showPicture(outpt,"keypointy");
}
vector<vector<float>> Fern::recognize(Mat picture,int *rec_point,int write,float* probability){
	
	if(write == 1)
		writeToFile(1);

	vector<KeyPoint> *processedKeyPoints	= getKeypoints(picture);
	Mat descriptors							= *getDescriptors(picture, processedKeyPoints);

	descriptors.convertTo(descriptors, CV_32FC1);
	vector<int> mapper(descriptors.rows);
	vector<vector<float>> resultVect(descriptors.rows, vector<float>(3));
	vector<int> recClassesVect;
	vector<int> keyPointsMatched;
	//inicializacia pola 
	for(int i=0; i<classes+1; i++){
		recClassesVect.push_back(0);
	}

	Mat result;
	long decimal;
	String des;
	String fewBites;
	int fernIndex;
	int recClass	=-1;
	int recDes		=-1;
	float max		= 0;
	int sumForDes	= 0;
	int position;
	vector<float> helpVector(classes,0);
	vector<int> fern_positions;
	printf("zacina rozpoznavanie");
	//prechadzanie kazdeho keypointu - deskriptoru z rozpoznavaneho objektu
	for(int point =0;point<descriptors.rows;point++){
		//inicializacia maxima, sumy a binarneho stringu, taktiez pre istotu sa nadstavit rozpoznana trieda a deskriptor na hodnotu -1
		des			= "";
		sumForDes	= 0;
		max			= 0;
		recDes		=-1;
		recClass	=-1;

		//*********************Vytvorenie binarneho retazca**************************//
		//vytvorenie binarneho retazca z deskriptoru
		for(int j=0;j<descriptors.cols;j++){

			float number = descriptors.at<float>(point,j);
			std::bitset<9> bs (number);
			des += bs.to_string();
		}
		//*********************Koniec tvorby binarneho retazca**************************//

		
		//*********************Samotne rozpoznavanie neznameho keypointu**********************//
		//cyklus na prejdenie prveho rozmeru histogramu - pocet natrenovanych tried
		for(int i=0;i<classes;i++){

			//cyklus na prechadzanie jednotlivych keypointov v natrenovanom histograme
			for(int j=0;j<histogram[i].size();j++){

				//pre kazdy keypoint(deskriptor) sa vynuluje hodnota sumy
				sumForDes = 0;
				//vyber fernov zo vstupneho retazca neznameho obrazka
				for(fernIndex=0;fernIndex<40;fernIndex++){
							
				fewBites	= des.substr(randomNumbers[fernIndex],FERN_SIZE);
				decimal		= binary_to_decimal(fewBites);
				position	= decimal+(pow(2,FERN_SIZE)*fernIndex);

				sumForDes	+= histogram[i][j][position];
				}

				if(sumForDes>max ){
					float fitness = ((float)sumForDes/float(40*ROTATION_COUNT))*100;

					//if(fitness > 15){
						max			= sumForDes;
						recClass	= i;
						recDes		= j;
						resultVect[point][0] = recClass;
						resultVect[point][1] = recDes;
						resultVect[point][2] = sumForDes;
					
				/*}
					else{
						recClass	= -1;
						recDes		= 0;
						resultVect[point][0] = -1;
						resultVect[point][1] = 0;
					}
					*/
					}
			}

		}
		//*********************Koniec rozpoznavania neznameho keypointu**********************//
		//po konci cyklusu sa najde keypoint na ktory sa neznamy keypoint namapuje
		
		mapper[point]	= recClass;
		if(recClass != -1){
			recClassesVect[recClass]++;
			helpVector[recClass] += resultVect[point][2];
		}
	}

	for(int i =0;i<recClassesVect.size()-1;i++){
		helpVector[i] /= recClassesVect[i];
	}

	float maxF = 0;
	int finalPoint = 0;
	for(int i=0;i<recClassesVect.size();i++){

		if(recClassesVect[i]>maxF){
			maxF			= recClassesVect[i];
			finalPoint		= i;
		}
	}
	*rec_point = finalPoint;
	*probability = maxF;
	return resultVect;
}

void showPicture(Mat picture,char *name){
	namedWindow(name);
	imshow(name,picture);
}

void Fern::train(Mat *picture,int number){
		rotate(picture,number);
}

vector <KeyPoint> rotateKeyPoints1(vector<KeyPoint> keypoints,int angle,Point center){
	//RotatedRect myRect;
	std::vector<cv::Point2f> points;
	Point2f onePoint;
	vector<KeyPoint>::iterator it;
	vector<KeyPoint> fKeyPoints;
	float newX,newY;
	float test,test2;
	angle = -angle;
	RotatedRect rect;

	cout << angle;

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

long binary_to_decimal(string num) /* Function to convert binary to dec */
{
    long dec = 0, n = 1, exp = 0;
    string bin = num;
       if(bin.length() > 1020){
          cout << "Binary Digit too large" << endl;
       }
       else {

            for(int i = bin.length() - 1; i > -1; i--)
            {
                 n = pow(2,exp++);
                 if(bin.at(i) == '1')
                 dec += n;
            }

       }
  return dec;
}

Fern::~Fern(void)
{
}
