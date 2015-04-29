
#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <opencv2/legacy/legacy.hpp>
#include "Classificator.h"
#include "Rotate.h"
#include <math.h>
#include "BinaryNaiveBayes.h"
#include <time.h>

using namespace cv;
using namespace std;
/**
* @function main
*TODO: mapovanie keypointov po rozpoznani - ransac nejako, ukladanie nahodnych fernov do zoznamu - .txt
*/
#define PI 3.14159265
clock_t start, endTime;
double cpu_time_used;

typedef struct imageInformation{
	Mat *descriptors;
	vector<KeyPoint> *keypoints;
	Mat *picture;
	Mat *binaryPatern;
	int cislo;
	int trieda;
	char name[100];
	imageInformation *next;
}ImageInformation;

//***********METHODS**************
Mat *loadTrainingSet();
void MatchPictures(ImageInformation image1, ImageInformation image2,int desMode, int mode);
vector<char*> listFile(char* path,bool isTrainingSet);
ImageInformation* computeKeyPoints(int mode, ImageInformation *imageInformation);
void bayes(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData);
void knn(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, int K);
float evaluate(cv::Mat& predicted, cv::Mat& actual);
Mat labelData(cv::Mat points, int equation);
int f(float x, float y, int equation);
void OLBP(Mat src,Mat *dst);
//void showPicture(Mat picture,char *name);
ImageInformation *getPicture(char* pathFile);
Mat *getLabels(Mat data);
void showLegend();
ImageInformation* getPictureFromTemplate(int number);
void rotate(ImageInformation *image,Classificator classif);
Mat getKeypoints(vector<KeyPoint> keypoints);
int getDiagonal(Mat *picture);
vector <KeyPoint> rotateKeyPoints(vector<KeyPoint> keypoints,int angle,Point center,Mat rotationMatrix);
void showHomography(vector<vector<float>> points, ImageInformation image1, ImageInformation image2,int recClass);
vector<KeyPoint> *getKeypointsOrig(Mat picture);
Mat *getDescriptorsOrig(Mat picture,vector<KeyPoint> *keyPoints);

//*************Global variables***********************
ImageInformation *firstImage = NULL;
int listCount = 0;
vector<char*> folders;
vector<char*> files;
vector<int> *triedy;
int KEY_POINT_MODE = 4;
bool wasTrained;
int maxTried = 0;

int main( int argc, const char** argv )
{


//************FIELDS**************
Mat pSocrates;
Mat pSocratesOrig;
Mat bluredSocrates;
Mat element;
ImageInformation *processedImage;
ImageInformation *akt;
Mat *binaryPatern = new Mat;
Mat *trainingData = new Mat;
Mat *testData = new Mat;

char* pathPrefix = "C:\\Pictures\\Nezname\\";
//****Koniec nacitavania obrazkov********
	akt = firstImage;

	//nacitanie folderov
	folders = listFile("C:\\Pictures\\PocitacoveVidenie\\",false);
	triedy = new vector<int>();
	trainingData = loadTrainingSet();
	trainingData->convertTo(*trainingData, CV_32FC1);

	//vytvorenie Mat ktora ma rovnaky pocet riadkov ako trenovaciaMnozina
	Mat* trainingClasses = getLabels(*trainingData);

	//******nacitanie neznameho prvku**********
	vector<char*> nezname = listFile(pathPrefix,false);
	
	//ImageInformation *unknown = getPicture(nezname.back());
	//pre testovanie zatial pridanie len jedneho Mat

	//showPicture(*unknown->picture,"Vstupny obrazok");
	//***************testing end*********************

	ImageInformation *aktual;
	aktual = firstImage;
	
	int *RecognizedClass = new int;
	vector<vector<float>> points;

	int const naive = 1;
	int const FileLoader = 0;
	
	
	Fern *fernStructure = new Fern(listCount);
	BinaryNaiveBayes *bNaiveBayes = new BinaryNaiveBayes();
	bNaiveBayes->init(listCount);
	
	if(FileLoader == 1){
		cout << "loading from file"<<endl;
		fernStructure->loadFromFile();
		cout << "loading from file"<<endl;
	}
	//Spustenie trenovania semi-naive bayes
		start = clock();
		for(int i=0;i<listCount;i++){
			fernStructure->train(aktual->picture,i);
			aktual->cislo = i;
			aktual = aktual->next;
		}
		endTime = clock();
		cpu_time_used = ((double) (endTime - start)) / CLOCKS_PER_SEC;
		cout << "trenovanie semi-naivneho bayesa trvalo: ";
		cout << cpu_time_used <<endl;

		cout << "trenovanie pomocou naivneho bayes";
		//spustenie trenovanie naive bayes
		aktual = firstImage;
		start = clock();
		for(int i=0;i<listCount;i++){
			bNaiveBayes->trainBayes(aktual->picture,i);
			aktual->cislo = i;
			aktual = aktual->next;
		}
		endTime = clock();
		cpu_time_used = ((double) (endTime - start)) / CLOCKS_PER_SEC;
		cout << "trenovanie naivneho bayesa trvalo: ";
		cout << cpu_time_used <<endl;
		ImageInformation *unknown;
		while(nezname.size()>0){
	
	unknown = getPicture(nezname.back());
	nezname.pop_back();
	//rozpoznanie pomocou semi-naive bayes
	start = clock();
	points = fernStructure->recognize(*unknown->picture,RecognizedClass,FileLoader);
	endTime = clock();
	cpu_time_used = ((double) (endTime - start)) / CLOCKS_PER_SEC;
	cout << "rozpoznavanie pomocou semi-naivneho bayesa trvalo: ";
	cout << cpu_time_used <<endl;

	int akt_p = 0;
	akt = firstImage;
	while(akt->cislo != *RecognizedClass){
		akt = akt->next;
	}

	showHomography(points,*unknown,*akt,*RecognizedClass);
	waitKey(0);

		
	//rozpoznanie pomocou naive-bayes
	cout << "rozpoznavanie pomocou naivneho bayes";
	start = clock();
	points = bNaiveBayes->findMatch(*unknown->picture,RecognizedClass);
	endTime = clock();
	cpu_time_used = ((double) (endTime - start)) / CLOCKS_PER_SEC;
	cout << "rozpoznavanie pomocou naivneho bayesa trvalo: ";
	cout << cpu_time_used <<endl;


	akt_p = 0;
	akt = firstImage;
	while(akt->cislo != *RecognizedClass){
		akt = akt->next;
	}

	showHomography(points,*unknown,*akt,*RecognizedClass);
	waitKey(0);
		}
	cout<<"koniec";

	
return 0; 
}

void showHomography(vector<vector<float>> points, ImageInformation unknown, ImageInformation recognized,int recClass){
	vector<Point2f> object, scene;
	  
 std::vector< DMatch > good_matches;
 Mat descriptor_object;
 Mat descriptor_scene;


 //BFMatcher matcher(NORM_L2);
vector<DMatch> matches;

for(int i=0;i<points.size();i++){
	if(points[i][0] == recClass)
	{
	DMatch match_point(points[i][1],i,recClass,points[i][2]);
		matches.push_back(match_point);
	}
}

//todo matches nadstavit
  try{
  //matcher.match( *recognized.descriptors,*unknown.descriptors, matches );
  }catch (Exception &e){
  cout << "chyba";
  }

   double max_dist = 0; double min_dist = 10000;

  //-- Quick calculation of max and min distances between keypoints
   /*
  for( int i = 0; i < recognized.descriptors->rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }
  
  for( int i = 0; i < recognized.descriptors->rows; i++ )
  { if( matches[i].distance < 3000*min_dist )
     { 
		 good_matches.push_back( matches[i]); }
  }
  */
  Mat img_matches;
  drawMatches( *recognized.picture, *recognized.keypoints,*unknown.picture, *unknown.keypoints, 
				matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  for(int i =0;i<matches.size();i++){
		scene.push_back(unknown.keypoints->at(matches.at(i).trainIdx).pt);
		object.push_back(recognized.keypoints->at(matches.at(i).queryIdx).pt);
 }
 
Mat H = findHomography( object, scene, CV_RANSAC );
Mat warpImage2;


std::vector<Point2f> unknown_corners;
std::vector<Point2f> recognized_corners;
Mat img_object = *unknown.picture;
 //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners,scene_corners, H);
  //imshow( "Good Matches & Object detection", warpImage2 );


  //line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  //line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  //line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  //line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );


  //-- Show detected matches
  resize(img_matches,img_matches,Size(1920*2/3,1080*2/3));
  imshow( "Good Matches", img_matches);

  waitKey(0);
}

vector<KeyPoint> *getKeypointsOrig(Mat picture){

	std::vector<cv::KeyPoint> *keypointsA = new vector<KeyPoint>;
	cv::Ptr<cv::FeatureDetector> detector;

	detector = cv::Algorithm::create<cv::FeatureDetector>("Feature2D.BRISK");
	detector->detect(picture, *keypointsA);
	return keypointsA;
}

Mat *getDescriptorsOrig(Mat picture,vector<KeyPoint> *keyPoints){
		Mat *descriptors = new Mat;
		cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;

			switch(KEY_POINT_MODE){
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

void rotate(ImageInformation *image,Fern *fern){
	Mat *picture = image->picture;
	vector<KeyPoint> *keypoints = image->keypoints;
	vector<KeyPoint> *newKeypoints = new vector<KeyPoint>;
	double angle = -1.0;
	double scale = 1;
	Mat new_picture;
	Mat help;
	Mat mKeypoints;
	Mat rotated_keypoints;

	String original = "original";
	String warped = " warped";
	help = *picture;
	
	int diagonal = getDiagonal(picture);
	
	Size newSize(diagonal,diagonal);

	Point center = Point( picture->size().width/2,picture->size().height/2);
	Mat rot_mat = getRotationMatrix2D( center, angle, scale );
	*newKeypoints = *image->keypoints;
	for(int i=0;i<360;i++){
	angle = rand() % 360;
	rot_mat = getRotationMatrix2D( center, angle, scale );
	warpAffine( *picture, new_picture, rot_mat, picture->size());
	
	*newKeypoints = rotateKeyPoints(*keypoints,angle,center,rot_mat);
	//warpAffine( mKeypoints, rotated_keypoints, rot_mat, mKeypoints.size() );

	//ShowKeyPoints(new_picture,0,newKeypoints);
	//drawKeypoints(const Mat& image, const vector<KeyPoint>& keypoints, Mat& outImage, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	//ako prerobit keypoints na Mat rotacia keypointov cez ten ity warpAffine a ukazat tie keypointy po rotacii, vykoncat vypocet deskriptorov po rotacii, zvacsit okno o diagonalu

	waitKey(0);
	}
}
int getDiagonal(Mat* picture){
	int newSize;

	int a = picture->rows;
	int b = picture->cols;

	newSize = sqrt(pow(a,2)+pow(b,2));
	return newSize;
}


vector <KeyPoint> rotateKeyPoints(vector<KeyPoint> keypoints,int angle,Point center,Mat M){
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
void showLegend(){
	ImageInformation *akt;
	akt = firstImage;
	while(akt->next!=NULL){
		printf("picture %s : class: %d\n",akt->name,akt->trieda);
		akt = akt->next;
	}
}
Mat* loadTrainingSet(){

		ImageInformation *akt;
		files = listFile("C:\\Pictures\\PocitacoveVidenie\\",true);
		//printf(files.size);
		vector<KeyPoint> *keypoints;
		Mat *data = new Mat;

		firstImage = (ImageInformation *) malloc(sizeof(ImageInformation));
			akt=firstImage;
	int number = 0;
    while(files.size()>0) {

		if(files.size() == 1) 
			printf("som tu");
		akt->binaryPatern = new Mat;
		akt->descriptors = new Mat;
		akt->keypoints = new vector<KeyPoint>();
		akt->picture = new Mat;
		akt->cislo = number++;
		akt->trieda = triedy->back();

		char* name = files.back();
		strcpy(akt->name,name);
		Mat picture = imread(name,CV_LOAD_IMAGE_GRAYSCALE);

		int diagonal = getDiagonal(&picture);

		Mat bigPicture = Mat(diagonal,diagonal, picture.type(), 1);

		cv::Rect roi( cv::Point( (diagonal - picture.cols)/2, (diagonal - picture.rows)/2 ), picture.size() );
		picture.copyTo( bigPicture( roi ) );

		*akt->picture = bigPicture;
		files.pop_back();
		triedy->pop_back();

		akt = computeKeyPoints(KEY_POINT_MODE,akt);
		data->push_back(*akt->descriptors);

		akt->next = (ImageInformation *) malloc(sizeof(ImageInformation));
		akt = akt->next;
	}
	akt->next = NULL;

		return data;
}

ImageInformation *getPicture(char* pathFile){

	ImageInformation *akt;
		
	akt = (ImageInformation *) malloc(sizeof(ImageInformation));

	//*****************Inicializacia struktury********************//
	akt->picture = new Mat;
	akt->descriptors = new Mat;
	akt->keypoints = new vector<KeyPoint>;
	akt->binaryPatern = new Mat;
	strcpy(akt->name,pathFile);

	(*akt->picture) = imread(pathFile,CV_LOAD_IMAGE_GRAYSCALE);

	//Vypocet keypointov a descriptorov pre obrazok
	akt->keypoints = getKeypointsOrig(*akt->picture);
	akt->descriptors = getDescriptorsOrig(*akt->picture,akt->keypoints);
	//OLBP(*akt->picture,akt->binaryPatern);

	return akt;
}


Mat createBinary(Mat picture)
{
	 //Grayscale matrix
    cv::Mat grayscaleMat (picture.size(), CV_8U);

    //Convert BGR to Gray
    cv::cvtColor( picture, grayscaleMat, CV_BGR2GRAY );

    cv::Mat binaryMat(grayscaleMat.size(), grayscaleMat.type());

    //Apply thresholding
    cv::threshold(grayscaleMat, binaryMat, 100, 255, cv::THRESH_BINARY);

    //Show the results
    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output", binaryMat);

	return binaryMat;
}

vector<char*> listFile(char* Path,bool trainingSet){
        DIR *pDIR;
		vector<char*> data;
        struct dirent *entry;
		int folderNumber=-1;
		char currentDirectory[200];
		char *name;

		if(trainingSet == true){
			while(folders.size()>0){
				strcpy(currentDirectory,folders.back());
				folders.pop_back();
				folderNumber++;

			if( pDIR=opendir(currentDirectory) ){
                while(entry = readdir(pDIR)){
					if(entry->d_type)
                        if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ){
                        cout << currentDirectory << entry->d_name << "\n";
						name = (char*)malloc(sizeof(char)*260);
						strcpy(name,currentDirectory);
						strcat(name,"\\");
						strcat(name,entry->d_name);
						data.push_back(name);
						triedy->push_back(folderNumber);
						listCount++;						
						}
                }
                closedir(pDIR);
			}
        }
		}

		else{
        if( pDIR=opendir(Path) ){
                while(entry = readdir(pDIR)){
					if(entry->d_type)
                        if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ){
                        cout << entry->d_name << "\n";
						name = (char*)malloc(sizeof(char)*260);
						strcpy(name,Path);
						strcat(name,entry->d_name);
						data.push_back(name);
						}
                }
                closedir(pDIR);
        }
		}
		return data;
	
}

Mat getDescriptors(ImageInformation *imageInformation){

	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
	Mat descriptors;

	descriptorExtractor =cv::Algorithm::create<cv::DescriptorExtractor>("Feature2D.BRIEF");
	descriptorExtractor->compute((*imageInformation->picture), (*imageInformation->keypoints), descriptors);

	return descriptors;
}
ImageInformation* computeKeyPoints(int mode, ImageInformation *imageInformation){

	//*****************Variables*********************/
	std::vector<cv::KeyPoint> keypointsA;
	cv::Mat descriptorsA;
	Mat outpt;
	cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;

	
	switch(KEY_POINT_MODE){
	case 1:{
			detector = cv::Algorithm::create<cv::FeatureDetector>("Feature2D.BRISK");
			descriptorExtractor =cv::Algorithm::create<cv::DescriptorExtractor>("Feature2D.BRIEF");
			break;
		   }

	case 2:{
			detector = cv::Algorithm::create<cv::FeatureDetector>("Feature2D.BRISK");
			descriptorExtractor =cv::Algorithm::create<cv::DescriptorExtractor>("Feature2D.BRISK");

			break;
		   }

	case 3:{
			detector = cv::Algorithm::create<cv::FeatureDetector>("Feature2D.BRISK");
			descriptorExtractor =cv::Algorithm::create<cv::DescriptorExtractor>("Feature2D.ORB");
			break;
		   }

	case 4:{
			detector = cv::Algorithm::create<cv::FeatureDetector>("Feature2D.BRISK");
			descriptorExtractor =cv::Algorithm::create<cv::DescriptorExtractor>("Feature2D.FREAK");
			break;
		   }
	}

	detector->detect((*imageInformation->picture), (*imageInformation->keypoints));

	//cv::drawKeypoints((*imageInformation->picture), (*imageInformation->keypoints), outpt, Scalar::all(10), DrawMatchesFlags::DEFAULT);

	descriptorExtractor->compute((*imageInformation->picture), (*imageInformation->keypoints), (*imageInformation->descriptors));

	return imageInformation;
}

void OLBP(Mat src,Mat *dst) {
		*dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            int center = src.at<unsigned char>(i,j);
            unsigned char code = 0;
            code |= (src.at<unsigned char>(i-1,j-1) > center) << 7;
            code |= (src.at<unsigned char>(i-1,j) > center) << 6;
            code |= (src.at<unsigned char>(i-1,j+1) > center) << 5;
            code |= (src.at<unsigned char>(i,j+1) > center) << 4;
			code |= (src.at<unsigned char>(i+1,j+1) > center) << 3;
            code |= (src.at<unsigned char>(i+1,j) > center) << 2;
            code |= (src.at<unsigned char>(i+1,j-1) > center) << 1;
            code |= (src.at<unsigned char>(i,j-1) > center) << 0;
            dst->at<unsigned char>(i-1,j-1) = code;
        }
    }
	printf("hotovo");
}

void MatchPictures(ImageInformation image1, ImageInformation image2, int desMode, int mode){

BruteForceMatcher<L2<float>> matcher;
CvNormalBayesClassifier bayes;
vector<DMatch> matches;
string windowName;
Mat imageMatches;
Mat response;
Mat result;

if(mode == 0){
matcher.match((*image1.descriptors),(*image2.descriptors), matches);

nth_element(matches.begin(), matches.begin()+24, matches.end());
matches.erase(matches.begin()+25, matches.end());

drawMatches(*image1.picture, *image1.keypoints, *image2.picture, *image2.keypoints, matches, imageMatches, Scalar(255,255,255));
}

if(desMode == 1){
	windowName = "BRIEF";
}
else if(desMode == 2){
	windowName = "BRISK";
}
else if(desMode == 3){
	windowName = "ORB";
}
else if(desMode == 4){
	windowName = "FREAK";
}

namedWindow(windowName);
imshow(windowName, imageMatches);
}

void knn(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, int K) {

    CvKNearest knn(trainingData, trainingClasses, cv::Mat(), false, K);

		int index,k,j;

	Mat *testClasses = new Mat(testData.rows, 1,CV_32S);
    cv::Mat predicted(testClasses->rows, 1, CV_32F);
    for(int i = 0; i < testData.rows; i++) {
            const cv::Mat sample = testData.row(i);
            predicted.at<float>(i,0) = knn.find_nearest(sample, K);
    }

    cout << "Accuracy_{KNN} = " << predicted << endl;
    //plot_binary(testData, predicted, "Predictions KNN");
}

ImageInformation* getPictureFromTemplate(int number){
	//ziskanie obrazka podla cisla z trenovacej mnoziny
	ImageInformation *akt;
	akt = firstImage;

	while(akt!=NULL){
		if(akt->cislo == number) return akt;
		akt = akt->next;
	}

	return akt;
}

void bayes(Mat& trainingData, Mat& trainingClasses, Mat& testData) {
	
	try{

		ImageInformation *akt = firstImage;
		CvNormalBayesClassifier bayes;// = new CvNormalBayesClassifier;
		Mat response;
		Mat *responseInt;
		Mat test,test2;
		Mat *result = new Mat;
		int j,k,resultValue,index;
	
		//skonvertovanie vstunej matice s trenovacimi obrazkami do float formatu
		
		bool returnValue = bayes.train(trainingData, trainingClasses, test, test2, false);

		resultValue = bayes.predict(testData,result);

		cout << "vysledok je trieda: " << resultValue;
		
		//showPicture(getPictureFromTemplate(resultValue),"vysledok pre Bayes");

	}catch(const std::exception& ex){
		cout << "error pri bayes";
	};
}

Mat *getLabels(Mat data){

	Mat *responseInt = new Mat(data.rows, 1,CV_32S);
	ImageInformation *akt = firstImage;
	int k,j;
		//vytvorenie matice ktora ma v sebe label-i ku vsetkym riadkom deskriptora
		int index = 0;
		int triedaCislo;
		printf("_______________________________________________________________________");
		while(akt->next!=NULL){
			triedaCislo = akt->trieda;
			if(triedaCislo > maxTried) maxTried = triedaCislo;
			printf("subor: %s a pridavaju sa cisla: %d\n",akt->name,triedaCislo);

			for(j=0;j<akt->descriptors->rows;j++){
				responseInt->at<int>(index,0) = triedaCislo;
				index++;
			}
			akt = akt->next;
		}
		return responseInt;
}
