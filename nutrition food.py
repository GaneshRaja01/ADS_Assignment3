# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:23:18 2023

@author: rgane
"""

# Importing the required libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt


def clean_df(df, time, class_name):
    """
    Cleans a pandas dataframe by filtering rows based on the specified time 
    and classification name.
    """
    df1 = df[df['Time'] == time]
    df2 = df1[df1['Classification Name'] == class_name]
    return df2


def remove_cols(df, cols):
    """
    Remove specified columns from a pandas DataFrame.
    """
    df.drop(cols, axis=1, inplace=True)
    return df


def line(x, m, c):
    """
    Compute the values of a linear function at the specified input points.
    """
    y = (m * x) + c
    return y


# main code
df = pd.read_excel('Food_Prices_for_Nutrition.xlsx')

# year for which clustering and fitting to be done
Time = 2017
class_name = 'Food Prices for Nutrition 1.1'

df2017 = clean_df(df, Time, class_name)

# columns to be removed
cols = ['Classification Name', 'Classification Code', 'Country Code',
        'Time', 'Time Code']
df17_cleaned = remove_cols(df2017, cols)

# replace .. with zeroes
df17_cleaned.replace('..', 0.0, inplace=True)

# set 'Country Name' as index and transpose the dataframe
df_t = df17_cleaned.set_index('Country Name').T

# Cluster start here
df_clus = df17_cleaned[["Cost of fruits [CoHD_f]",
                        "Cost of vegetables [CoHD_v]"]].copy()

# normalise dataframe and inspect result using cluster tools
df_clus, df_min, df_max = ct.scaler(df_clus)
print(df_clus.describe())

print("n   score using silhouette score")
# loop over trial numbers of clusters calculating the silhouette
for ic in range(2, 10):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_clus)

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(df_clus, labels))

# display the clusters in the plot
nc = 6  # number of cluster centres

kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(df_clus)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

# display the plot
plt.figure(figsize=(6.0, 6.0))
plt.scatter(df_clus["Cost of fruits [CoHD_f]"],
            df_clus["Cost of vegetables [CoHD_v]"], c=labels, cmap="tab10")

# show cluster centres
xc = cen[:, 0]
yc = cen[:, 1]
plt.scatter(xc, yc, c="red", marker="d", s=80)

#label and title
plt.xlabel("Cost of fruits [CoHD_f]")
plt.ylabel("Cost of vegetables [CoHD_v]")
plt.title("6 clusters")
plt.show()


# fitting code
x = df17_cleaned['Affordability of a nutrient adequate diet: ratio of cost to the food poverty line [CoNA_pov]']
y = df17_cleaned['Affordability of a healthy diet: ratio of cost to the food poverty line [CoHD_pov]']

# curve fit
popt, pcorr = opt.curve_fit(line, x, y)

# extract variances and sigma
sigmas = np.sqrt(np.diag(pcorr))

z = line(x, *popt)

# display the fit
plt.figure()
plt.title("Linear")
plt.scatter(x, y, label="data")
plt.plot(x, z, label="fit")

plt.legend(loc="upper left")
plt.show()