from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import warnings
import matplotlib.pyplot as plt
from FUNCS import plot_k_means_cluster, find_clusters, get_file
from mpl_toolkits.mplot3d import Axes3D

def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.filterwarnings('ignore')
k_means = KMeans(init="k-means++", n_clusters = 4, n_init = 12)
# clusters = find_clusters(k_means)
# clusters.show()
kmeans_plot = plot_k_means_cluster(k_means)
# kmeans_plot.show()

url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv'
pathname = 'Cust_Segmentation.csv'
cust_df = get_file(url, pathname)
# print(cust_df.head())
df = cust_df.drop('Address', axis=1)
df.head()

X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
# print(labels)

df["Clus_km"] = labels
# print(df.groupby('Clus_km').mean())
# print("X.shape: ", X.shape)
# print("X: ", X)
# for x in X:
#     print(x)
area = np.pi * ( X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

# plt.show()


fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
plt.show()
















