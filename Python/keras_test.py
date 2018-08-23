from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet #has promise
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
# load the model
#model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
model = MobileNet(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# load an image from file
test_file_dir = 'C:\\Users\\mitch\\Desktop\\MovementImg\\'
import os
from os import listdir
openNames = []
files = os.listdir(test_file_dir)
for x in files:
    if os.path.isfile(test_file_dir + x):
        openNames.append(test_file_dir + str(x))

image = load_img(openNames[len(openNames)-1], target_size=(224, 224))
print("loaded image #",len(openNames)-1)
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
labelPredict = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = labelPredict[0][0]
label2 = labelPredict[0][1]
label3 = labelPredict[0][2]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))
print('%s (%.2f%%)' % (label2[1], label2[2]*100))
print('%s (%.2f%%)' % (label3[1], label3[2]*100))
openFile = open("C://Users//mitch//Desktop//ImgText//guess.txt", 'w')
outString = str(label[1]) + "," + str(label[2])
print(outString)
print(outString, file = openFile)
openFile.close()
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

