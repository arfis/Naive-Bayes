package test;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

/*
 * Getting the input picture from the external device - webcam and throwing it to output
 */
public class Camera {
	static{ System.loadLibrary("opencv_java249"); }
	static MatOfByte matOfByte;
	static JFrame frame = new JFrame();
	static JLabel label = new JLabel();
	static Mat kniha;
	static Keypoint img_k;
	static Classifier classif = new Classifier();
	static Fern fern = new Fern(200);
	
    public Camera (Keypoint k){
    
    matOfByte = new MatOfByte();
    System.out.println("Hello, OpenCV");
    // Load the native library.
    k.init_camera(this);
	img_k = k;
    }
    
    public void openCamera()
    {
    	
    final VideoCapture camera = new VideoCapture(0);

    if(!camera.isOpened()){
        System.out.println("Camera Error");
    }
    else{
        System.out.println("Camera OK?");
    }
   
    //thread for capturing images from camera
    new Thread()
    {
        public void run() 
       {
        	try {
				while (System.in.available() == 0) {
				Mat frame = new Mat();	
				
				camera.read(frame);
				//transformation of the input image to grayscale
				Imgproc.cvtColor(frame, frame,Imgproc.COLOR_RGB2GRAY);
				fern = new Fern(200);
				fern.train(img_k.Extract(frame));
				fern.showHistogram();
				Thread.sleep(2000);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
        	camera.release();
        }
    }.start();
} 
    



//drawing the image into output
public static void showResult(Mat img) {
	
    Imgproc.resize(img, img, new Size(640, 480));
    if(!img.empty())
    Highgui.imencode(".jpg", img, matOfByte);
  
    byte[] byteArray = matOfByte.toArray();
    BufferedImage bufImage = null;
    
    try {
        InputStream in = new ByteArrayInputStream(byteArray);
        bufImage = ImageIO.read(in);
        label.setIcon(new ImageIcon(bufImage));
        frame.getContentPane().add(label);
   
        frame.pack();
        frame.setVisible(true);
    } catch (Exception e) {
        e.printStackTrace();
    }
    
}


}