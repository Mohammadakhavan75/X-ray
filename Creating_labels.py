import pandas as pd
import pickle

all_xray_df = pd.read_csv('E:\\Work\\Kaggle\\ChexNet\\Data_Entry_2017.csv')
my_dict = {all_xray_df['Image Index'][i]: all_xray_df['Finding Labels'][i] for i in range(len(all_xray_df))}
my_dict2 = {all_xray_df['Image Index'][i]: my_dict[all_xray_df['Image Index'][i]].split('|') for i in
            range(len(all_xray_df))}
disease = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
           'Pneumothorax', 'Atelectasis', 'Fibrosis', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']
num_disease = [i for i in range(15)]
my_dict3 = {disease[i]: num_disease[i] for i in range(15)}
temp = [0 for i in range(15)]
labels = []

for i in range(len(all_xray_df)):
    for j in range(len(my_dict2[all_xray_df['Image Index'][i]])):
        temp[my_dict3[my_dict2[all_xray_df['Image Index'][i]][j]]] = 1
    labels.append(temp)
    temp = [0 for i in range(15)]

my_dict4 = {all_xray_df['Image Index'][i]: labels[i] for i in range(len(all_xray_df))}
f = open('lables_dict.p', 'wb')
pickle.dump(my_dict, f)
f.close()
