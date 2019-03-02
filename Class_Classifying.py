import pandas as pd
import os
import glob

data = pd.read_csv('C:\\Users\\hpc\\Desktop\\MMD\\ChexNet\\Dataset\\Data_Entry_2017.csv')
img_path = os.path.join('C:\\Users\\hpc\\Desktop\\MMD\\ChexNet\\Dataset\\images')

disease = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumothorax', 'Atelectasis', 'Fibrosis', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']

listarr = []
for d in disease:
    k=0
    arr = []
    for i in range(len(data)):
        if(data['Finding Labels'][i]==d and k<51):
            k=k+1
            arr.append(data['Image Index'][i])
    listarr.append(arr)
