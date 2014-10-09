package test;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;

public class Matrix
{
   public static void main( String[] args )
   {
	  System.loadLibrary("opencv_java249");
      System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
      System.out.println("Welcome to OpenCV " + Core.VERSION);
      Classifier clas = new Classifier();
      Keypoint k = new Keypoint();
      Camera cam = new Camera(k);
      cam.openCamera();
      LoadTrainingSet Loader = new LoadTrainingSet(cam,k,clas);
      Loader.openFiles();
      
    
   }
   
}
