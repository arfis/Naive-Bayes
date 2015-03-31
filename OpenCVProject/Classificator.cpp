#include "stdafx.h"
#include "Classificator.h"
#include <unordered_map>
#include <math.h>

using namespace std;
using namespace cv;

typedef struct word
{
	String name;
	Fern *f;
	word *next;
}names;

int SIZE = 0;
unordered_map<String,Fern> mapa;
names *nam;
Fern *f1,*f2;
names *first = NULL;
int learned = 0;
int fernS = 16;
Fern *learnedFern;

float equation(float average, float var,int value);
//vector<int> sumHist(Fern* f);
vector<int> calculatePosterior(Fern *unknown);
void check();

Classificator::Classificator(void)
{
	
}


Classificator::~Classificator(void)
{
}

void Classificator::train(Mat trainingData, Mat trainingClasses,int max)
{
	max += 1;
	learnedFern = new Fern(trainingData, trainingClasses,max);

}
int Classificator::predict(Mat data){

	//learnedFern->identifyFern(data);
	
//Fern *learnedFern;
	learnedFern->loadUnknown(data);
	compute(learnedFern);
	return 0;
}

vector<int> Classificator::identify(Fern *f,int mode)
{
	
	return calculatePosterior(f);
	
	
}

vector<int> Classificator::compute(Fern *input){

	vector<int> recog(input->numberOfClasses);
	vector<vector<int>> ferns;

	for(int i=0;i<pow(2,8);i++){
		//ferns = input->trainedHistogram;
		
		for(int i =0;i<input->numberOfClasses;i++){
			recog[i] += input->dataHistogram[i][input->trainedHistogram[i]];
		}
	}
	printf("prva hodnota je: %d \n druha hodnota je: %d", recog[0], recog[1]);


	waitKey(0);
return recog;

}
	//the method for making the sum of ferns in the histogram and finding the point that has the biggest probability
/*
vector<int> sumHist(Fern *trained)
{
	int *histRec;
	int max = 0;
	int sum = 0;
	int lear;
	String fernNazov;
	int keypoint = 0;
	Fern *akt;
	names *ak;
	vector< vector <int> > akt_hist;
	vector<vector<int>> unk_hist = unkFern->getHistogram();
	vector<int> recog;

	try{
	histRec = (int *)calloc(2,sizeof(int));
	//chosing the size of the input unknown vector
	for(int column = 0;column<unkFern->getSize1();column++)
	{
	ak = first;
	max = 0;
	//the number of learned samples	
	for(int num_learned = 0;num_learned<learned;num_learned++)
	{
		akt = ak->f;
		akt_hist = akt->getHistogram();
			//going through the histogram array rows with the image from the training set
			for(int j=0;j<akt->getSize1();j++)
			{
				sum = 0;
				//suming the columns for every keypoint based on the input unknown fern
				for(int k=0;k<unkFern->getSize2();k++)
				{
					//modulo cause the sum number can be greater than the size of the akt_hist - fernSize
					sum += akt_hist[j][unk_hist[column][k]%16];
				}
				//if the sum is greater then the max, change the max and get the name of the fern - name of the picture and the keypoint that
				//it should be

				if(sum>max)
				{
					max = sum;
					fernNazov = akt->getName();
					keypoint = j;
					lear = num_learned;
					recog.push_back(lear);
				}
			}
			ak = ak->next;
	}
	unk_hist[column].push_back(keypoint);
	unk_hist[column].push_back(lear);
	//printf("%d %d\n", lear + 1, keypoint);
	histRec[lear]++;
	}
	}catch(Exception e){
		e.formatMessage();};
	cout << "Naive Bayes\n";
	for(int i = 0;i<2;i++)
	{
		cout << histRec[i];
		cout << " ";
	}
	return recog;
}
*/
void check()
{
	names *akt = first;
	printf("checking: \n");
	for(int i =0;i<learned;i++)
	{
		printf("average: \n");
		akt->f->showAverage();
		akt = akt->next;
		//cout << akt->f->getTest();
	}

	cout << "OTHER TEST";
	//f1->calculateVariance();
	//f1->showAverage();
	//f2->calculateVariance();
	//f2->showAverage();
}
vector<int> calculatePosterior(Fern *unknown)
{
vector< vector <int> > uhist;
vector <int> rec_post;
int *HistRec;
float *tvar;
float *tavg;
float post;
float post_max=0;
int max = 0;
int imageno = 0;
names *akt = first;
Fern *a;

uhist = unknown->getHistogram();
HistRec = (int *)calloc(2,sizeof(int));

//computing for every keypoint from the unknown - which image suits the best
for(int column = 0; column<unknown->getSize1();column++)
{
	post = 1;
	post_max = 0;
	akt = first;
	
for(int j=1;j<=learned;j++)
{
	a = akt->f;
	tvar = a->getVar();
	tavg = a->getAverage();
	//a->showAverage();
	post = 1;
	for(int i=0;i<fernS;i++)
	{
		post = post * equation(tvar[i],tavg[i],uhist[column][i]);
	}
	if(post > post_max)
	{
			post_max = post;
			imageno = j;
	}
	//next fern in the trained queue
	akt = akt->next;
}
uhist.at(column).push_back(imageno);
HistRec[imageno-1]++;
rec_post.push_back(imageno);
}
cout << "Ordinary Bayes\n";
for(int i =0;i<2;i++)
{
	printf("%d ",HistRec[i]);
}
cout << "\n";
return rec_post;
}

float equation(float var, float average,int value)
{
	float sum;

	float sq = (sqrt(2*3.14*var));
	float ex = -(pow(value-average,2))/(2*var);
	sum = 1/sq*exp(ex);
	return sum;
}