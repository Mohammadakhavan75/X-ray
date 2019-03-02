import pickle
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np

os.chdir('E:\\Work\\Kaggle\\ChexNet\\Arrays')

labels = pickle.load(open('labels_dict.p', 'rb'))

input_shape=(224,224,1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(8192, (1, 1), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(15, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

epochs = 100
batch_size = 32

for i in range(epochs):
    for j in range(49):
        #datas = pickle.load(open('dict' + str(j) + '.p', 'rb'))
        with open('dict' + str(j)+ '.p', 'rb') as f:
            datas = pickle.load(f, encoding='latin1')

        keys = datas.keys()

        x_train = []
        y_train = []
        for key in keys:
            x_train.append(datas[key])
            y_train.append(labels[key])
        x_train = np.asarray(x_train)
        model.fit(x_train, y_train, batch_size=batch_size, verbose=1)
    print("epoch: "+str(epochs))
