# -------------------------------------------------------------------------
# AUTHOR: Joey Cindass
# FILENAME: clustering_turnin.py
# SPECIFICATION: checking k values on which maximizes Silhouette coefficient
# FOR: CS 4210- Assignment #5
# TIME SPENT: 3 hours
# -----------------------------------------------------------*/

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('training_data.csv', sep=',', header=None)

X_training = df.values

max_sil_sc = 0
sil_coefs = []

for k in range(2, 21):
    print(k)
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(X_training)
    temp_sil_coef = silhouette_score(X_training, kmeans.labels_)
    sil_coefs.append(temp_sil_coef)
    if silhouette_score(X_training, kmeans.labels_) > temp_sil_coef:
        max_sil_sc = temp_sil_coef


plt.plot(range(2, 21), sil_coefs)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Coefficients')
plt.title("Silhouette Coefficient vs. Number of Clusters")
plt.show()

df = pd.read_csv('testing_data.csv', sep=',', header=None)

labels = np.array(df.values).reshape(1, df.shape[0])[0]

print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
