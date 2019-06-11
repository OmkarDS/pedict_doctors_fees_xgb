import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re

#Percentage of missing values
#print(round(df.isnull().sum()/len(df)*100,2))
df = pd.read_excel("Final_Train.xlsx")

'''
Cleaning Qualification col
'''

df['Qual_1'] = df['Qualification'].str.split(',').str[0]
df['Qual_2'] = df['Qualification'].str.split(',').str[1]
df['Qual_3'] = df['Qualification'].str.split(',').str[2]
df['Qual_1'].fillna("XXX",inplace = True)
df['Qual_2'].fillna("XXX",inplace = True)
df['Qual_3'].fillna("XXX",inplace = True)

df = df.drop(['Qualification'], axis = 1)

le_qual_1 = LabelEncoder()
le_qual_2 = LabelEncoder()
le_qual_3 = LabelEncoder()

df['Qual_1'] = le_qual_1.fit_transform(df['Qual_1'])
df['Qual_2'] = le_qual_2.fit_transform(df['Qual_2'])
df['Qual_3'] = le_qual_3.fit_transform(df['Qual_3'])
