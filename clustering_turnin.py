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

X_training = []
for i in df:
    temp = []
    for j in df:
        temp.append(j)
    X_training.append(temp)
print(X_training)

# for i in df:
#     X_training.append(i)
#

max_sil_sc = 0
best_k = 0
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)
    # kmeans.fit(X_training)
    # print(kmeans.labels_)
    sil_sc = silhouette_score(X_training, kmeans.labels_)
    if sil_sc > max_sil_sc:
        max_sil_sc = sil_sc
        best_k = k

# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X_training)
# labels = kmeans.labels_
# print(X_training)
# silhouette_score(X_training, labels)
# print(labels)
# kmeans.fit(X_training)
# print(kmeans.labels_)
# s_avg = silhouette_score(X_training, labels)
