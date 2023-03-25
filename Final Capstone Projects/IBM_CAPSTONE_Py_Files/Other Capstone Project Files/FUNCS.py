import requests as rs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.datasets import make_blobs


def get_df(filename):
    return pd.read_csv(filename)

def download(url, filename):
    response = rs.get(url)
    if response.status_code == 200:
        # print(response.content)
        with open(filename, "wb") as f:
            f.write(response.content)
            
def get_file(url, filename):
    if filename is None:
        raise ValueError("Filename Must Be Specified and cannot be None")
    try:
        file = open(filename)
        file.close()
    except FileNotFoundError:
        if url is None:
            raise ValueError("There is no file with the given name and the url is not specified.")
        download(url, filename)
    return get_df(filename)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Returns the plot figure object. .show() may be called to see
    the accompanying graph.
    This function prints confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


def decision_boundary(X, y, model, iris, two=None, plot_colors = "ryb", plot_step = 0.02):
    """ Returns a plt object and plt.show() may be called on the return object."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    
    if two:
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        for i, color in zip(np.unique(y), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], label=y, cmap=plt.cm.RdYlBu, s=15)
        plt.show()
    
    else:
        set_ = {0, 1, 2}
        print(set_)
        for i, color in zip(range(3), plot_colors):
            idx = np.where(y == i)
            if np.any(idx):
                set_.remove(i)
                
                plt.scatter(X[idx, 0], X[idx, 1], label=y, cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
        
        for i in set_:
            idx = np.where(iris.target == i)
            plt.scatter(X[idx, 0], X[idx, 1], marker='x', color='black')
        
        return plt
    
    
def plot_probability_array(X,probability_array):
    """ Returns a plt object and plt.show() may be called on the return object."""

    zeros = np.zeros((X.shape[0], 30))
    plot_array=zeros
    col_start=0
    
    for class_,col_end in enumerate([10,20,30]):
        plot_array[:,col_start:col_end]= np.repeat(probability_array[:,class_].reshape(-1,1), 10,axis=1)
        col_start=col_end
    plt.imshow(plot_array)
    plt.xticks([])
    plt.ylabel("samples")
    plt.xlabel("probability of 3 classes")
    plt.colorbar()
    return plt


def find_clusters(k_means, starting_centroids=[[4, 4], [-2, -1], [2, -3], [1, 1]]):
    np.random.seed(0)
    X, y = make_blobs(n_samples=5000, centers=starting_centroids, cluster_std=0.9)
    plt.scatter(X[:, 0], X[:, 1], marker='.')
    
    k_means.fit(X)
    k_means_labels = k_means.labels_
    print(k_means_labels)
    return plt


def plot_k_means_cluster(k_means, dims=(6, 4), starting_centroids=[[4, 4], [-2, -1], [2, -3], [1, 1]]):
    # Initialize the plot with the specified dimensions.
    fig = plt.figure(figsize=dims)
    
    X, y = make_blobs(n_samples=5000, centers=starting_centroids, cluster_std=0.9)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_labels
    # Colors uses a color map, which will produce an array of colors based on
    # the number of labels there are. We use set(k_means_labels) to get the
    # unique labels.
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
    
    # Create a plot
    ax = fig.add_subplot(1, 1, 1)
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_cluster_centers
    
    # For loop that plots the data points and centroids.
    # k will range from 0-3, which will match the possible clusters that each
    # data point is in.
    for k, col in zip(range(len(starting_centroids)), colors):
        # print("K: ", k)
        # print("COL: ", col)
        # Create a list of all data points, where the data points that are
        # in the cluster (ex. cluster 0) are labeled as true, else they are
        # labeled as false.
        my_members = (k_means_labels == k)
        
        # Define the centroid, or cluster center.
        cluster_center = k_means_cluster_centers[k]
        
        # Plots the datapoints with color col.
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
        
        # Plots the centroids with specified color, but with a darker outline
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    
    # Title of the plot
    ax.set_title('KMeans')
    
    # Remove x-axis ticks
    ax.set_xticks(())
    
    # Remove y-axis ticks
    ax.set_yticks(())
    return plt