import pandas as pd
import keras
import pickle

data = pd.read_csv('E:\\Work\\Kaggle\\ChexNet\\Data_Entry_2017.csv')

disease = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
           'Pneumothorax', 'Atelectasis', 'Fibrosis', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']

origtemp = []
for i in range(len(data)):
    temp = []
    litemp = data['Finding Labels'][i].split('|')
    for j in range(len(litemp)):
        for k in range(len(disease)):
            if litemp[j] == disease[k]:
                temp.append(k)
    temp = keras.utils.to_categorical(temp, 15)
    lastemp = [0 for i in range(15)]
    for h in range(len(temp)):
        lastemp += temp[h]
    origtemp.append([data['Image Index'][i], lastemp])
my_dict = {origtemp[i][0]: origtemp[i][1] for i in range(len(origtemp))}

f = open('labels.p', 'wb')
pickle.dump(my_dict, f)
f.close()
