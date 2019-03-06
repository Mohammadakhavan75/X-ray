from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from matplotlib import pyplot as plt
import numpy as np
import pickle
import random

#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1


input_img = Input(shape=(224, 224, 1))
x = Conv2D(512, (3, 3), activation='relu', padding='same')(input_img)
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
#x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
#x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
#x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='sigmoid', padding='same')(x)
encoded = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
#x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
#x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
#x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='Adamax', loss='mean_squared_error')

#with open('../Dicts/dict0.p','rb') as f:
#	datas = pickle.load(f, encoding='latinl')
#
#keys = datas.keys()
##print(keys)
#temp = random.sample(keys, 1)
#x_train = datas[temp[0]]
#x_train = np.asarray(x_train)
#x_train = np.reshape(x_train, (1, 224, 224, 1))
#
#from keras.callbacks import TensorBoard 
#
#autoencoder.fit(x_train, x_train, epochs=50, callbacks=[TensorBoard()])
#
for j in range(10):
    with open('../Dicts/dict' + str(j) + '.p', 'rb') as f:
        datas = pickle.load(f, encoding='latin1')

    keys = datas.keys()

    x_train = []
    x_test = []
    for key in keys:
        x_train.append(datas[key])
        x_test.append(datas[key])
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

print(len(x_train))
x_train = np.reshape(x_train[:100], (len(x_train[:100]), 224, 224, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test[:100], (len(x_test[:100]), 224, 224, 1))  # adapt this if using `channels_first` image data format

from keras.callbacks import TensorBoard

autoencoder.fit(x_train[:100], x_train[:100], epochs=50, batch_size=128, shuffle=True, validation_data=(x_test, x_test), callbacks=[TensorBoard()])

#decoded_imgs = autoencoder.predict(x_test)

# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
