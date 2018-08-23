import numpy as np
import tensorflow as tf
import time
import cv2



class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    
        # load an image from file
    test_file_dir = 'C:\\Users\\mitch\\Desktop\\MovementImg\\'
    import os
    from os import listdir
    openNames = []
    files = os.listdir(test_file_dir)
    for x in files:
        if os.path.isfile(test_file_dir + x):
            openNames.append(test_file_dir + str(x))

    model_path = 'C:\\Users\\mitch\\AppData\\Local\\Programs\\Python\\Python36\\faster_rcnn_inception_v2_coco_2018_01_28\\frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    image = cv2.imread(openNames[len(openNames)-1], 1);
    img = cv2.resize(image, (100, 100))
    oneFound = False
    boxes, scores, classes, num = odapi.processFrame(img)
    maximum = np.max(scores)
    for i in range(len(boxes)):
        # Class 1 represents human
        if(scores[i] == maximum):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                print("Human")
                oneFound = True
            if classes[i] == 18 and scores[i]> threshold:
                print("Dog")
                oneFound = True
            if classes[i] == 17 and scores[i]> threshold:
                print("Cat")
                oneFound = True
            if classes[i] == 16 and scores[i]> threshold:
                print("Bird")
                oneFound = True
            if classes[i] == 3 and scores[i]> threshold:
                print("Car")
                oneFound = True
    if(not oneFound):
        print("No Known Objects Detected")
    #Wipe files after running

    import os, shutil
    folder = 'C:\\Users\\mitch\\Desktop\\MovementImg\\'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    
    
    



