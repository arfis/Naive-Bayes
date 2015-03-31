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

//TO DO: -add to the histogram a new column which tells that from which picture it is
//-adding all keypoints to one big two dimensional array
using namespace std;


//Ksize is the number of keypoints - number of rows in the histogram
//fernSize when used as 2^fernSize, number of columns - fern size number of bites it takes from the binary descriptor string

int no;
//int Ksize;
long binary_to_decimal(string num);
vector<int> normalize(vector<int> vect,int cols);

vector< vector <int> > *dataHistogram = new vector<vector<int>>();
vector<int> *trainedHistogram = new vector<int>();
cv::String name;
void calculateAverage();
int test = 0;
int numberOfClasses;

void setTest(int i)
{
	test = 3;
}

int getTest()
{
	return test;
}

Fern::Fern()
{
}

void Fern::printVariance()
{
	printf("printing variance");
	for(int i =0; i<16; i++)
	{
		cout << variance[i];
		cout << "\n";
	}
}

void Fern::showAverage()
{
	int i;
	for(i=0;i<16;i++)
	{
		cout << average[i];
		cout << "\n";
	}
}

float* Fern::getAverage()
{
	return average;
}

float* Fern::getVar()
{
	return variance;
}

void Fern::calculateVariance()
{
float var;
calculateAverage();
//sum of every column
for(int i=0;i<fernSize;i++)
{
	var = 0;
	for(int j = 0; j<Ksize; j++)
	{
		var += (average[i] - dataHistogram[j][i])*(average[i] - dataHistogram[j][i]);
	}

	var /= Ksize-1;
	variance[i] = pow(var,2);
}
}

void Fern::loadUnknown(Mat data){
	String des;
	String fewBites;
	std::vector<int> histogram(pow(2,8));

	Mat result;
	long decimal;

	for(int i =0;i<data.rows;i++){
		for(int j=0;j<data.cols;j++){

			float number = data.at<float>(i,j);
			
			std::bitset<8> bs (number);
			des = bs.to_string();

			for(int binaryIndex=0;binaryIndex<8;binaryIndex++){
				fewBites = des.substr(binaryIndex,8);
				decimal = binary_to_decimal(fewBites);
				binaryIndex = binaryIndex + 7;
				histogram[(int)decimal]++;
			}
	  }
	}
	
	trainedHistogram = normalize(histogram,(int)pow(2,8));
	printf("end creating unknown ferns");
	
}

vector<vector<int>> normalize(vector<vector<int>> vect,int rows,int cols){
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			vect[i][j] %= 256;
		}
	}
	return vect;	
}

vector<int> normalize(vector<int> vect,int cols){
		for(int j=0;j<cols;j++){
			vect[j] %= 255;
		}
	return vect;	
}

Fern::Fern(Mat data,Mat classes,int classesInt)
{
	//vytovrenie histogramu pre vsetky triedy
	//vector<vector<int>> *histogram(classesInt, vector<int>(pow(2,8));
	String des;
	String fewBites;
	std::vector<std::vector<int>> histogram(
    classesInt,
    std::vector<int>(pow(2,8)));
	Mat result;
	long decimal;

	numberOfClasses = classesInt;

	for(int i =0;i<data.rows;i++){
		for(int j=0;j<data.cols;j++){

			float number = data.at<float>(i,j);
			
			std::bitset<8> bs (number);
			des = bs.to_string();

			for(int binaryIndex=0;binaryIndex<8;binaryIndex++){
				fewBites = des.substr(binaryIndex,8);
				decimal = binary_to_decimal(fewBites);
				binaryIndex = binaryIndex + 7;
				histogram[classes.at<int>(i)][(int)decimal]++;
			}
	  }
	}
	//histogram = normalize(histogram,classesInt, pow(2,8));
	dataHistogram = histogram;
	printf("end");
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

String Fern::getName()
{
return name;
}

void Fern::create(cv::String binary)
{
int index;
int number;
no++;
//creating fern from received binary string -taking just 3 bite, the max value they can have is 16
for(index=-1;index<(int)binary.size()-1;)
{
	//for(int i =0;i<fernSize;i++){
	number = 0;
	number += binary[++index]-'0';
	number += (binary[++index]-'0')*2;
	number += (binary[++index]-'0')*4;
	number += (binary[++index]-'0')*8;
	
	//adding the computed number to the histogram
	dataHistogram[no][number]++;
	//if(histogram[no][number] > 100) printf("vysokaaa hodnota!");
}
}

void Fern::calculateAverage()
{
	for(int i =0;i<fernSize;i++)
	{
	//creating average
		for(int j = 0;j<Ksize;j++)
		{
			average[i] += dataHistogram[j][i];
		}
	average[i] /= Ksize;
	}
}

void Fern::showHistogram()
{
	int index1,index2;
	for(index1=0;index1<Ksize;index1++)
	{
		for(index2=0;index2<dataHistogram.at(index1).size();index2++)
		{
			cout << dataHistogram.at(index1).at(index2);
			cout << " ";
		}
		cout << "\n";
	}
}

vector< vector <int> > Fern::getHistogram()
{
	return dataHistogram;
}

int Fern::getSize1()
{
	return Ksize;
}

int Fern::getSize2()
{
	return fernSize;
}