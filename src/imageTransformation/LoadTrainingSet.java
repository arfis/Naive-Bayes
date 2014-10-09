package imageTransformation;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import Classification.Camera;
import Classification.Classifier;
import Classification.Fern;
import Classification.Keypoint;

public class LoadTrainingSet {
	Mat[] trainingData;
	MatOfKeyPoint[] keyPoints = new MatOfKeyPoint[3];
	//the number of pictures in the training set
	int number = 2;
	Keypoint keypoint;
	Classifier clas;
	Camera cam;
	//the maximum number received from the keypoint
	Fern fern[] = new Fern[200];
	
	public LoadTrainingSet(Camera c,Keypoint k,Classifier clas)
	{
		trainingData = new Mat[number];
		this.keypoint = k;
		this.clas = clas;
		this.cam = c;
	}
	
	public void openFiles()
	   {
		  
				for(int i=1;i<=number;i++)
				{
					Creator c = new Creator();
					System.out.println("loading traning set");
					 trainingData[i-1] = Highgui.imread("C:\\Users\\Michal\\workspace\\Recognition\\TrainSet\\"+i+".jpg",Highgui.CV_LOAD_IMAGE_COLOR);
					 Imgproc.cvtColor(trainingData[i-1], trainingData[i-1], Imgproc.COLOR_RGB2GRAY );
					 
					 trainingData[i-1] = c.Rotate(trainingData[i-1]);
				}
				System.out.println("identifying keypoints");
				IdentifyKeypoints();
	   }
	
	public void IdentifyKeypoints()
	{
		for(int i=0;i<number;i++)
		{
			System.out.println("identifikacia keypointov");
			fern[i] = new Fern(200);
			System.out.println("Keypointy na vzorke");
			if (trainingData[i].empty()) break;
			keypoint.init_camera(cam);
			keyPoints[i] = keypoint.Extract(trainingData[i]);
			fern[i].train(keyPoints[i]);
			fern[i].setName("Chobotnica");
			//fern[i].showHistogram();
			fern[i].getPercentage();
		}
		
	}
	  
}
