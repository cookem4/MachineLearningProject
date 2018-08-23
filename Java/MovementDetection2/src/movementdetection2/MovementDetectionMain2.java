package movementdetection2;

import org.opencv.core.Core;


public class MovementDetectionMain2 {
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		//new RunVideo2().run();
		new RunVideo1().run();
		//new RunVideo3().run();
	}
}
