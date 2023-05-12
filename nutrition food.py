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
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt


def clean_df(df, time, class_name):
    df1 = df[df['Time'] == time]
    df2 = df1[df1['Classification Name'] == class_name]
    return df2


def remove_cols(df, cols):
    df.drop(cols, axis=1, inplace = True)
    return df


def line(x, m, c):
    y = m*x+c
    return y


# main code
df = pd.read_excel('Food_Prices_for_Nutrition.xlsx')

Time = 2017
class_name = 'Food Prices for Nutrition 1.1'

df2017 = clean_df(df, Time, class_name)

cols =['Classification Name', 'Classification Code', 'Country Code', 'Time', 'Time Code']
df17_cleaned = remove_cols(df2017, cols)
df17_cleaned.replace('..' , 0.0, inplace=True)



######################
df_clus = df17_cleaned[["Cost of fruits [CoHD_f]", "Cost of vegetables [CoHD_v]"]].copy()

# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_clus to affect df17_cleaned. This make the plots with the 
# original measurements
df_clus, df_min, df_max = ct.scaler(df_clus)
print(df_clus.describe())
print()

print("n   score")
# loop over trial numbers of clusters calculating the silhouette
for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_clus)     

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_clus, labels))


nc = 6 # number of cluster centres

kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(df_clus)     

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))
# scatter plot with colours selected using the cluster numbers
plt.scatter(df_clus["Cost of fruits [CoHD_f]"], df_clus["Cost of vegetables [CoHD_v]"], c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours

# show cluster centres
xc = cen[:,0]
yc = cen[:,1]
plt.scatter(xc, yc, c="k", marker="d", s=80)
# c = colour, s = size

# plt.xlabel("total length")
# plt.ylabel("height")
# plt.title("4 clusters")
plt.show()


#############fitting
x = df17_cleaned['Affordability of a nutrient adequate diet: ratio of cost to the food poverty line [CoNA_pov]']
y = df17_cleaned['Affordability of a healthy diet: ratio of cost to the food poverty line [CoHD_pov]']
popt, pcorr = opt.curve_fit(line, x, y)

# extract variances and calculate sigmas
sigmas = np.sqrt(np.diag(pcorr))

# call function to calculate upper and lower limits with extrapolation
# create extended year range

z= line(x, *popt)


plt.figure()
plt.title("Linear")
plt.scatter(x, y, label="data")
plt.plot(x, z, label="fit")
# plot error ranges with transparency

plt.legend(loc="upper left")
plt.show()
