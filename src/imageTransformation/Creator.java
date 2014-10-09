package imageTransformation;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class Creator {
	
	public Mat Rotate(Mat image){
	
		double radians = Math.toRadians(40);
		double sin = Math.abs(Math.sin(radians));
		double cos = Math.abs(Math.cos(radians));

		int newWidth = (int) (image.width() * cos + image.height() * sin);
		int newHeight = (int) (image.width() * sin + image.height() * cos);

		int[] newWidthHeight = {newWidth, newHeight};
		
		int pivotX = newWidthHeight[0]/2; 
		int pivotY = newWidthHeight[1]/2;
		
		org.opencv.core.Point center = new org.opencv.core.Point(pivotX, pivotY);
		Size targetSize = new Size(newWidthHeight[0], newWidthHeight[1]);
		
		Mat targetMat = new Mat(targetSize, image.type());

		int offsetX = (newWidthHeight[0] - image.width()) / 2;
		int offsetY = (newWidthHeight[1] - image.height()) / 2;
		

		Mat waterSubmat = targetMat.submat(offsetY, offsetY + image.height(), offsetX, offsetX + image.width());
		image.copyTo(waterSubmat);

		Mat rotImage = Imgproc.getRotationMatrix2D(center, 160, 1.0);
		Mat resultMat = new Mat(); // CUBIC
		Imgproc.warpAffine(targetMat, resultMat, rotImage, resultMat.size());
		return resultMat;
		
	}
}
