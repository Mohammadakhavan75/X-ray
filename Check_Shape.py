import pickle
import cv2
import imageio
import numpy as np

for j in range(11):
	with open('../Dicts/dict' + str(j)+ '.p', 'rb') as f:
		datas = pickle.load(f, encoding='latin1')

	keys = datas.keys()
	for key in keys:
		if datas[key].shape != (224, 224):
			a = datas[key][:,:,1]
			for i in range(len(a)):
				for j in range(len(a[i])):
					if a[i,j] != datas[key][i,j,1]:
						print('False!')


#with open('../Dicts/dict' + str(1)+ '.p', 'rb') as f:
#	datas = pickle.load(f, encoding='latin1')
#keys = datas.keys()
#print(keys)
#for key in keys:
#	print(key,datas[key].shape()) 
#print(datas['00001304_005.png'].shape)
