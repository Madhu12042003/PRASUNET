import pandas as pd
from sklearn.preprocessing import LabelBinarizer as lb
from tensorflow.keras.preprocessing.image import ImageDataGenerator as idg
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout,BatchNormalization
import os
import signal
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

training_data = pd.read_csv("C:\\MadhuPES\\SEMESTER6\\deepLearning\\sign_lang\\sign_lang\\train_dataset\\datasets\\sign_mnist_train.csv") # read training dataset into a dataframe
testing_data = pd.read_csv("C:\\MadhuPES\\SEMESTER6\\deepLearning\\sign_lang\\sign_lang\\testdata\\testdata\\sign_mnist_test.csv") # read testing dataset into a dataframe

# split labels and data
training_labels = training_data['label'] # isolate labels
testing_labels = testing_data['label'] # isolate labels

training_data = training_data.drop(['label'],axis=1) # drop labels from training dataset
testing_data = testing_data.drop(['label'],axis=1) # drop labels from testing dataset

X_train = training_data.values.reshape(-1, 28, 28, 1) # training data for model
X_test = testing_data.values.reshape(-1, 28, 28, 1) # testing data for model
X_test = X_test/255 # normalize values between 0 and 1

binarizer = lb() # use label binarizer to convert all class labels into binary
y_train = binarizer.fit_transform(training_labels) # training labels for model
y_test = binarizer.fit_transform(testing_labels) # testing labels for model

# to create augmented images for each image in dataset with different variations
augmented_image_gen = idg(rescale = 1./255, rotation_range = 3, height_shift_range=0.2, width_shift_range=0.2, horizontal_flip=False, zoom_range = 0.2, fill_mode='nearest')


# CNN model
CNN = Sequential() # to build a linear neural network
CNN.add(Conv2D(32,kernel_size=(6,6),strides=1,activation='relu',padding='same',input_shape=(28,28,1))) # Convolution Layer
CNN.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))                        # Max pooling Layer
CNN.add(Conv2D(64,kernel_size=(4,4),strides=1,activation='relu',padding='same'))            # Convolution Layer
CNN.add(MaxPool2D(pool_size=(2,2),strides=1,padding='same'))                        # Max pooling Layer
CNN.add(Conv2D(64,kernel_size=(3,3),strides=1,activation='relu',padding='same'))            # Convolution Layer
CNN.add(MaxPool2D(pool_size=(2,2),strides=2,padding='same'))                        # Max pooling Layer
CNN.add(Conv2D(128,kernel_size=(2,2),strides=1,activation='relu',padding='same'))           # Convolution Layer
CNN.add(MaxPool2D(pool_size=(2,2),strides=2,padding='same'))                        # Max pooling Layer
CNN.add(Flatten())  # flatten

CNN.add(Dense(units=512,activation='relu')) # Neural Network
CNN.add(Dropout(rate=0.25))
CNN.add(Dense(units=24,activation='softmax')) # Neural Network with softmax activation

CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
CNN.fit(augmented_image_gen.flow(X_train,y_train,batch_size=100), epochs = 3, validation_data=(X_test,y_test), shuffle=1) # fit model
CNN.save("C:\\MadhuPES\\SEMESTER6\\deepLearning\\sign_lang\\sign_lang\\model.h5") # save model 

final_training_loss, final_training_accuracy = CNN.evaluate(X_train, y_train, verbose=0)
final_testing_loss, final_testing_accuracy = CNN.evaluate(X_test, y_test, verbose=0)

print("Final Training Loss:", final_training_loss)
print("Final Training Accuracy:", final_training_accuracy)
print("Final Testing Loss:", final_testing_loss)
print("Final Testing Accuracy:", final_testing_accuracy)