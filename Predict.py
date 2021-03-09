import tensorflow as tf 
import os 
from cv2 import cv2  

model= tf.keras.models.load_model("cats_vs_dogs")

IMG_SIZE = 50

CATEGORIES = ["Cat","Dog"]

def create(file):

	img_array = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
	new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
	return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

predictions = model.predict([create("cat.jpg")])	

print(CATEGORIES[int(predictions[0])])

predictions = model.predict([create("dog.jpg")])	

print(CATEGORIES[int(predictions[0])])

	












	
