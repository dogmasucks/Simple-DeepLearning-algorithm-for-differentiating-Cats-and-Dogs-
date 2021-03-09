
import numpy as np
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


filepath = "E:/PetImages"
classes = ["Cat", "Dog"]

IMG_SIZE = 50


training_data = []


def create_training_data():
    for category in classes:
        
        path = os.path.join(filepath, category)  
        
        class_num = classes.index(category)

        for img in tqdm(os.listdir(path)):

            
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE) 
               
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                
                training_data.append([new_array, class_num])
            except Exception as e:  
                pass


create_training_data()

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

X = X/255.0

dense_layers = [0, 1, 2, 3, 4]
layer_sizes = [32, 64, 128,240]
conv_layers = [1, 2, 3, 4, 5]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:             #optimizing process
        for conv_layer in conv_layers:
                            
            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=(50, 50, 1)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for i in range(conv_layer-1):

                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

           
            model.add(Flatten())

            for l in range(dense_layer):

                model.add(Dense(64))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(
                log_dir="logs/dense_layer{}layer_size{}conv_layer{}".format(dense_layer, layer_size, conv_layer))
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            model.fit(X, y, batch_size=5, epochs=1, callbacks=[tensorboard],validation_split =0.3)

model.save("cats_vs_dogs_models")
