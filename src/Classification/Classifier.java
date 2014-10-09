package Classification;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvNormalBayesClassifier;

public class Classifier {

	CvNormalBayesClassifier classifier;
	
	public Classifier()
	{
		
		classifier = new CvNormalBayesClassifier();
	}
	public void train(Mat trainingData)
	{	
		Mat Keypoints = new Mat();
		Keypoints.create( trainingData.size(), CvType.CV_32FC1 );
		System.out.println(Keypoints.channels());
		//trainingData.convertTo(Keypoints, CvType.CV_32FC1);
		System.out.println(Keypoints.channels());
		classifier.train(Keypoints,Keypoints);
		//classifier.train
	}
	
}
