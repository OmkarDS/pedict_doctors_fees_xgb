import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re

def cleaning(filename, train = True):
    df = pd.read_excel(filename)

    #Percentage of missing values
    #print(round(df.isnull().sum()/len(df)*100,2))
    
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
    
    
    '''
    Cleaning Experience column
    
    '''
    exp_pattern = re.compile('(\d*)')
    df['Experience'] = df['Experience'].apply(lambda row : float(re.match(exp_pattern, row).group(1)))
    
    
    '''
    Cleaning Rating column
    '''
    
    df['Rating'] = df['Rating'].fillna('0%')
    rating_pattern = re.compile('(\d*)')
    df['Rating'] = df['Rating'].apply(lambda row : float(re.match(rating_pattern, row).group(1)))
    
    
    '''
    Cleaning Place column
    '''
    df['location'] = df['Place'].str.split(',').str[0]
    df['location'] = df['location'].fillna('unknown')
    df['city'] = df['Place'].str.split(',').str[1]
    df['city'] = df['city'].fillna('unknown')
    
    
    #'''
    #Visualization
    #'''
    #
    #sns.pointplot(data=df,
    #              y='Fees',
    #              x='city',
    #              capsize=.1)
    #plt.show()
    #
    #sns.barplot(data=df,y='Fees',x='city', hue = 'Profile')
    #plt.legend(loc = 'upper right')
    #plt.xticks(rotation = 40)
    #plt.show()
    
    le_location = LabelEncoder()
    le_city = LabelEncoder()
    
    df['location'] = le_location.fit_transform(df['location'])
    df['city'] = le_city.fit_transform(df['city'])
    
    df = df.drop(['Place'], axis = 1)
    
    '''
    Cleaning Miscellaneous_Info column
    '''
    misc_info_pattern = re.compile('((\d*)\%\s(\d*))')
    
    df['Miscellaneous_Info'] = df['Miscellaneous_Info'].fillna('unknown')
    df['no_of_ratings'] = df['Miscellaneous_Info'].apply(lambda row : 0 if re.match(misc_info_pattern, row)== None
      else (re.match(misc_info_pattern, row).group(3)))
    df['no_of_ratings'] = pd.to_numeric(df['no_of_ratings'])
    df = df.drop(['Miscellaneous_Info'], axis = 1)
    
    '''
    Cleaning Profile column
    '''
    
    le_profile = LabelEncoder()
    
    df['Profile'] = le_profile.fit_transform(df['Profile'])
    
    if train:
        X = df.drop(['Fees'], axis = 1).values
        y = df['Fees'].values
        
        X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        return df,X_train , X_test, y_train, y_test, X, y
    else:
        X = df.values
        return df, X