# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:23:18 2023

@author: rgane
"""

# Importing the required libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 


def clean_df(df, time, class_name):
    df1 = df[df['Time'] == time]
    df2 = df1[df1['Classification Name'] == class_name]
    return df2


def remove_cols(df, cols):
    df.drop(cols, axis=1, inplace = True)
    return df

# main code
df = pd.read_excel('Food_Prices_for_Nutrition.xlsx')

Time = 2017
class_name = 'Food Prices for Nutrition 1.1'

df2017 = clean_df(df, Time, class_name)

cols =['Classification Name', 'Classification Code', 'Country Code', 'Time', 'Time Code']
df17_cleaned = remove_cols(df2017, cols)
x = df17_cleaned.replace('..' , 0.0)
#df17_cleaned.replace('..', 0)
#df17_cleaned.fillna(0)
