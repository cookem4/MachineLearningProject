package movementdetection2;

import java.awt.Color;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.video.BackgroundSubtractorKNN;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;

public class RunVideo1 {
	Mat frame = new Mat();
	Mat subtractionMask = new Mat();
	VideoCapture camera = new VideoCapture(0);
	BackgroundSubtractorKNN mog = Video.createBackgroundSubtractorKNN();
	boolean objectFound = false;
	long startTime;
	long endTime;
	int imgNum = 0;
	boolean writeImg = false;
	Mat subSubMat = new Mat();
	//String cmdString = "C:\\Users\\mitch\\AppData\\Local\\Programs\\Python\\Python36\\python.exe C:\\Users\\mitch\\AppData\\Local\\Programs\\Python\\Python36\\keras_test.py";
	String cmdString = "C:\\Users\\mitch\\AppData\\Local\\Programs\\Python\\Python36\\python.exe C:\\Users\\mitch\\AppData\\Local\\Programs\\Python\\Python36\\keras_testV3.py";

	public volatile boolean runningPy = false;
	public volatile boolean doneFirstRun = false;
	boolean goGreen = false;
	String prediction = "";
	boolean humanNotFound = false;


	void run() {
		while(true) {
			if(!camera.read(frame)) {
				break;
			}
			else if (frame!=null){
				Imgproc.blur(frame, frame, new Size(5,5));
				//Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
				mog.apply(frame, subtractionMask, 0.3);
				Imgproc.blur(subtractionMask, subtractionMask, new Size(5,5));
				Imgproc.threshold(subtractionMask, subtractionMask, 254, 255, Imgproc.THRESH_BINARY);
				/*
				//Contours method
				List<MatOfPoint> contours = new ArrayList<>();
				Mat hierarchy = new Mat();
				Imgproc.findContours(subtractionMask,contours,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
				
				if (!hierarchy.empty()) {
					for (int i = 0; i >=0; i = (int)hierarchy.get(i, 0)[1]) {
						Moments moment = new Moments();
						List<Moments> nu = new ArrayList<Moments>(contours.size());
					    nu.add(0, Imgproc.moments(contours.get(i),false));
					    moment=nu.get(0);
					   
					    double area = moment.m00;
					    if(area > 10) {
					    	System.out.println(moment.m10/area + ", " + moment.m01/area);
					    	objectFound = true;
					    }else {objectFound = false;}
					}
					if(objectFound == true) {
						List<Point> rectCenters = new ArrayList();
						List<Rect> boundRect = new ArrayList();
						for(int k = 0; k < contours.size();k++) {
							boundRect.add(Imgproc.boundingRect(contours.get(k)));
						}
						for(int j = 0; j < boundRect.size(); j++) {
							//Imgproc.rectangle(frame, boundRect.get(j).tl(), boundRect.get(j).br(), new Scalar(0,0,255), 10);
							rectCenters.add(new Point((boundRect.get(j).tl().x + boundRect.get(j).br().x)/2, (boundRect.get(j).tl().y + boundRect.get(j).br().y)/2));
						}
						//Draw rectangle around centre poitns of bounding boxes of contours
						int maxX = 0;
						int maxY = 0;
						int minX = 100000;
						int minY = 100000;
						for(int j = 0; j < rectCenters.size();j++) {
							if(rectCenters.get(j).x > maxX) {
								maxX = (int) rectCenters.get(j).x;
							}
							if(rectCenters.get(j).y > maxY) {
								maxY = (int) rectCenters.get(j).y;
							}
							if(rectCenters.get(j).x < minX) {
								minX = (int) rectCenters.get(j).x;
							}
							if(rectCenters.get(j).y < minY) {
								minY = (int) rectCenters.get(j).y;
							}
						}
						Imgproc.rectangle(frame, new Point(minX, minY), new Point(maxX, maxY), new Scalar(0,255,0), 10);
						Imgproc.drawContours(frame, contours,0, new Scalar(0,0,255));
						
						System.out.println("FOUND");
					}
				}
				*/
				//Boudning rect from binary image method
				MatOfPoint points = new MatOfPoint();
				Core.findNonZero(subtractionMask, points);
				Rect boundRect = Imgproc.boundingRect(points);
				Mat submat = frame.submat(boundRect);
				if(!submat.empty() && submat.width()>50 && submat.height()>50) {
					if(humanNotFound) {
						Imgproc.rectangle(frame, boundRect.tl(), boundRect.br(), new Scalar(0,255,255), 10);
					}
					else {
						if(goGreen) {
							Imgproc.rectangle(frame, boundRect.tl(), boundRect.br(), new Scalar(0,255,0), 10);
						}
						else {
							Imgproc.rectangle(frame, boundRect.tl(), boundRect.br(), new Scalar(0,0,255), 10);
						}
					}
				
				}
				if(writeImg && !submat.empty() && submat.width()>50 && submat.height()>50) {
					imgNum++;
					System.out.println("WRITING");
					Rect tempRect = new Rect(new Point(6,6), new Point(submat.width()-5,submat.height()-5));					
					subSubMat = submat.submat(tempRect);
					Imgcodecs.imwrite("C://Users//mitch/Desktop//MovementImg//Image" + Integer.toString(imgNum) + ".jpg", subSubMat);
					
					if(runningPy == false) {
						callConsoleOnce();
					}
					
					writeImg = false;
				}
				if(startTime == 0) {
					startTime = System.currentTimeMillis();
				}
				if((endTime = System.currentTimeMillis()) - startTime  > 1000){
					writeImg = true;
					startTime = 0;
					endTime = 0;
				}
				/*
				if(doneFirstRun && !runningPy) {
					String nameProb = getName();
				}
				*/
				if(prediction.length()!=0) {
					if(prediction.length() > 24) {
						Imgproc.putText(frame, prediction, new Point(20,40), Core.FONT_HERSHEY_COMPLEX, 1, new Scalar(0,255,255));
						humanNotFound = true;
					}
					else {
						 Imgproc.putText(frame, prediction, new Point(20,40), Core.FONT_HERSHEY_COMPLEX, 1, new Scalar(0,255,0));
						 humanNotFound = false;
					}
					//Imgproc.putText(frame, prediction, new Point(20,40), Core.FONT_HERSHEY_COMPLEX, 1, new Scalar(0,255,0));
					goGreen = true;
				}
				else {
					Imgproc.putText(frame, "Predicting...", new Point(20,40), Core.FONT_HERSHEY_COMPLEX, 1, new Scalar(0,0,255));
				}
				HighGui.imshow("Window", frame);
				//HighGui.imshow("Mask", subtractionMask);
				if(submat.width() > 5 && submat.height() > 5 && !subSubMat.empty()) {
					Rect tempRect = new Rect(new Point(5,5), new Point(submat.width()-5,submat.height()-5));					
					subSubMat = submat.submat(tempRect);
					//HighGui.imshow("Movement", submat);
					//HighGui.resizeWindow("Movement", 500, 500);
				}
				HighGui.waitKey(1);
				
			}
			
		}
		//Clears Camera and closes window
		camera.release();
		HighGui.destroyAllWindows();
		
	}
	String getName() {
		String name = "";
		File folder = new File("C://Users//mitch//Desktop//ImgText//");
		File[] listOfFiles = folder.listFiles();
		if(listOfFiles.length > 0) {
			BufferedReader reader = null;

			try {
			    reader = new BufferedReader(new FileReader(listOfFiles[0]));
			    String text = null;

			    while ((text = reader.readLine()) != null) {
			        name = text;
			        System.out.println(name);
			    }
			} catch (FileNotFoundException e) {
			    e.printStackTrace();
			} catch (IOException e) {
			    e.printStackTrace();
			} finally {
			    try {
			        if (reader != null) {
			            reader.close();
			        }
			    } catch (IOException e) {
			    }
			}

		}
		return name;
	}
	void callConsoleOnce() {
		runningPy = true;
		try {
		final Process p = Runtime.getRuntime().exec("cmd /c" + cmdString);
		new Thread(new Runnable() {
		    public void run() {
		     BufferedReader input = new BufferedReader(new InputStreamReader(p.getInputStream()));
		     String line = null; 
		     int lineCount = 0;
		     try {
		        while ((line = input.readLine()) != null) {
		        	lineCount++;
		        	if(lineCount == 1) {
		        		prediction = line;
		        	}
		            System.out.println(line);
		            runningPy = false;
		        }
		        
		     } catch (IOException e) {
		            e.printStackTrace();
		     }
		    }
		}).start();
		}
		catch(Exception e) {}

		
	}
	public class myThread extends Thread{
		@Override
		public void run() {
			
			try {
				runningPy = true;
				System.out.println("pre execute");
				//Process p = Runtime.getRuntime().exec("python keras_test.py");
				Process p  =new ProcessBuilder("python", "keras_test.py").inheritIO().start();
				//ProcessBuilder pb = new ProcessBuilder("python", "keras_test.py");
				//Process p = pb.start();
		        BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
		        System.out.println(in.readLine());
				System.out.println("post execute");
				doneFirstRun = true;
				runningPy = false;
			}
			catch(Exception e) {
				System.out.println(e);
			}
		}
	}
}