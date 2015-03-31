
#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2\opencv.hpp>
#include<stdio.h>
#include<cstdlib>
#include<string.h>
#include<fstream>
#include<dirent.h>
#include <vector>
#include <opencv2/legacy/legacy.hpp>
#include "Classificator.h"

using namespace cv;
using namespace std;
/**
* @function main
*/

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
void showPicture(Mat picture,char *name);
ImageInformation *getPicture(char* pathFile);
Mat *getLabels(Mat data);
void showLegend();
ImageInformation* getPictureFromTemplate(int number);

//*************Global variables***********************
ImageInformation *firstImage = NULL;
int listCount = 0;
vector<char*> folders;
vector<char*> files;
vector<int> *triedy;
int KEY_POINT_MODE = 1;
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
	ImageInformation *unknown = getPicture(nezname.back());

	//pre testovanie zatial pridanie len jedneho Mat
	//Mat test = getPictureFromTemplate(11);
	testData->push_back(*unknown->descriptors);
	showPicture(*unknown->picture,"Vstupny obrazok");
	//***************testing end*********************
	//waitKey(0);

	//trainingClasses->convertTo(*trainingClasses,CV_32FC1);
	testData->convertTo(*testData, CV_32FC1);

	bayes(*trainingData, *trainingClasses,  *testData);
	//();
	//waitKey(0);
	//knn(*trainingData, *trainingClasses,  *testData,4);
	
	//**********Naive Bayes*************
	Classificator *classif = new Classificator();
	//classif->learn(NULL);
	classif->train(*trainingData,*trainingClasses,maxTried);
	classif->predict(*testData);
	
	//classif->learn(trainingData,trainingClasses);
//waitKey(0);

return 0; 
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

		*akt->picture = imread(name,CV_LOAD_IMAGE_GRAYSCALE);
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
	akt = computeKeyPoints(KEY_POINT_MODE,akt);
	OLBP(*akt->picture,akt->binaryPatern);

	return akt;
}

void showPicture(Mat picture,char *name){
	namedWindow(name);
	imshow(name,picture);
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

ImageInformation* computeKeyPoints(int mode, ImageInformation *imageInformation){

	//*****************Variables*********************/
	std::vector<cv::KeyPoint> keypointsA;
	cv::Mat descriptorsA;
	Mat outpt;
	cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;

	
	switch(mode){
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

	cv::drawKeypoints((*imageInformation->picture), (*imageInformation->keypoints), outpt, Scalar::all(10), DrawMatchesFlags::DEFAULT);

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
