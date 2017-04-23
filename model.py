import csv
import cv2
import numpy as np

lines=[]

with open('data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
	       lines.append(line)
images = []
measurements = []

#reading in the images using the csv file
lines=lines[1:len(lines)]
for line in lines:
    #looping for all the left center and right camera images
    for i in range(3):
        source_path=line[i]
        filename= source_path.split('/')[-1]
        current_path='data/IMG/'+filename
        image=cv2.imread(current_path)
        images.append(image)
    measurment= float(line[3])
    measurements.append(measurment)
    #adding and subtracting 0.3 for adjusting for the left and right camera images
    measurements.append(measurment+0.3)
    measurements.append(measurment-0.3)


X_train =np.array(images)
y_train =np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D


model= Sequential()

#Lambda layer for normalizing the training data
model.add(Lambda(lambda x: x/255.0-0.5,input_shape=(160,320,3)))
#Cropping the images to exclude the part that is not required
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(64,3,3, activation="elu"))
model.add(Convolution2D(64,3,3, activation="elu"))
#addded droput layer to prevent overfitting
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Using the adam optimizer and mean squared error as the loss function
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')
