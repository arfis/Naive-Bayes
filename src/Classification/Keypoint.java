package Classification;

import java.util.List;
import java.util.Vector;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Size;
import org.opencv.features2d.*;
import org.opencv.imgproc.Imgproc;

public class Keypoint {
	
	/*
	 * This class orients on keypoint extraction from the image
	 */
	
	private Mat mRgba;
	Mat descriptors, descriptors_kniha;           
	List<Mat> descriptorsList;
	FeatureDetector featureDetector;
	MatOfKeyPoint keyPoints = new MatOfKeyPoint();
	MatOfKeyPoint keyPoints_kniha = new MatOfKeyPoint();
	Camera cam;
	DescriptorMatcher descriptorMatcher;
	DescriptorExtractor briefDescriptor; 
	Vector<Mat> briefDescriptors;
	
	public Keypoint(Camera cam){
		this.cam = cam;
		featureDetector=FeatureDetector.create(FeatureDetector.FAST); //fast
		briefDescriptor = DescriptorExtractor.create(DescriptorExtractor.SURF);
		briefDescriptors = new Vector<Mat>();
	}
	
	public Keypoint(){
		
		featureDetector=FeatureDetector.create(FeatureDetector.FAST); //fast
		briefDescriptor = DescriptorExtractor.create(DescriptorExtractor.BRIEF);
		briefDescriptors = new Vector<Mat>();
	}
	public void init_camera(Camera c)
	{
		this.cam = c;
	}
	
	//keypoint detection of the input image
@SuppressWarnings("static-access")
public MatOfKeyPoint Extract(Mat image)
{

Size size = image.size();
Imgproc.GaussianBlur(image, image, new Size (3,3), 1.2, 1);
mRgba = new Mat(size, CvType.CV_8UC1);
	
descriptors = new Mat();
descriptors_kniha = new Mat();
image.copyTo(mRgba);

featureDetector.detect(mRgba, keyPoints);

briefDescriptor.compute(mRgba, keyPoints, descriptors);

Features2d.drawKeypoints(mRgba,keyPoints,image);
if(!image.empty())
cam.showResult(image);

return keyPoints;
}

}