# MachineLearningProject
A smart security camera software with the ability to detect and highlight moving objects in a given frame then classify the image.


#################################################################################################################################
1.OpenCV libraries are required to run the java project
2.There are 3 different methods of image classification employed from deep neural networks:
    - Keras image detection is used in the first module to return the image's label and the percent certainty of the guess
    - Faster RCNN is used to classify weather or not a human appears within the moving portion of the image
    - Faster RCNN is used to classify the moving portion of the image into one of several categories and return it
3. Java project uses the python classification modules to get predict image type. Done by inputting the portion of the image that is moving
   which is fileterd out through the openCV libraries
