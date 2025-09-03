# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Reading Spotify data. 
df = pd.read_csv(r'C:\Users\Chavez\Documents\School Content\2023 Fall\INSY 5377 - Web & Social Analytics\Data and Analysis\spotify-2023-project-data-clean.csv', encoding='Latin-1')

# Selecting numeric song traits to cluster
song_traits = ['bpm','danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

data = df[song_traits].copy()

# Produces summary statistic numbers and saves it to a CSV file in the current directory
summary_statistics = data.describe()
summary_statistics.to_csv('spotify-2023-summary-statistics.csv', header=True)

#1 Scale the data so each column gets treated equally
#2 Initializing random centroids first
#3 Label each data point based on how far the data point is from each centroid for the cluster assignment for each song
#4 Update centroids and look at each song and look at their labels and find the center points
#5 Repeat steps #3 and #4 until the centroids stop changing


# SCALING THE DATA
# Setting the minimum value in each column to zero by subtracting the minimum value from each item then divide by the range to rescale everything from zero to one.
# Multiplying by 9 rescales everything from 0 to 9.
# Adding 1 at the end to shift from 0 to 9 with a 1 to 10 scale.
data = ((data - data.min()) / (data.max() - data.min())) * 9 + 1

# Double check the min and max values are 0 and 10, respectively.
print(data.describe())
summary_cluster_statistics = data.describe()
summary_cluster_statistics.to_csv('spotify-2023-normalized-summary-statistics.csv', header=True)


# INITIALIZING RANDOM CENTROIDS
# The apply method is iterating through each column in the data, selects a single value from that column, a random value, for each column. Then convert panda series to float.
# Initializing a list called centroids
def random_centroids(data, k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    # combines all the individual panda series into a data frame
    return pd.concat(centroids, axis=1)

centroids = random_centroids(data, 3)
print(centroids)

# Label each data point based on how far the data point is from each centroid for the cluster assignment for each song
# Calculating the distance between each centroid and each data point and then finding the cluster assignment for each song.
def get_labels(data, centroids):
    # Evaluating the distance between the first data point and evaluate the distance between the data point's centroid and assign it to the nearest cluster by calculating the distance.
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    # Finds the index of the minim value which would also be the cluster assignment
    return distances.idxmin(axis=1)

labels = get_labels(data, centroids)

# Value_counts will count how many times each unique value occurs in a column
labels.value_counts()

# UPDATING CENTROIDS BASED ON WHICH SONGS ARE IN THE CLUSTER
# Finding all the songs in a cluster and then taking the geometric mean of each song trait
# First splitting the data frame that has our song trait data, aplitting it up by cluster, the label series gives us the series assignments, each group we are applying the function that calculated the geometric mean of the centroids to get our new centroids.
def new_centroids(data, labels, k):
    centroids = data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids


# Principle Component Analysis (PCA) will look at our song trait columns and summarize them.
# matplotlib will do the plotting portion
# clear_output will help clear the output each time we plot a new graph


def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}')
    
    # Plot data points with different colors for each cluster
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_points = data_2d[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')

    # Plot centroids
    plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1], marker='X', s=200, c='black', label='Centroids')
    
    # Display legend
    plt.legend()
    plt.show()

max_iterations = 100 # The max number of iterations unless the cluster stopped changing
centroid_count = 3 # Number of clusters

centroids = random_centroids(data, centroid_count)
old_centroids = pd.DataFrame() # Will be stopping the algorithm once the centroids stop changing
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    
    labels = get_labels(data, centroids)
    centroids = new_centroids(data, labels, centroid_count)
    plot_clusters(data, labels, centroids, iteration)
    iteration += 1

# Saving and printing general centroid contents
print(centroids)
centroids.to_csv('spotify-2023-centroids.csv', header=True)

# Saving and printing cluster 0 contents and summary statistics and saves it to a CSV file in the current directory
cluster_0_contents = df[labels ==0]
cluster_0_contents.to_csv('spotify-2023-cluster_0_contents.csv', header=True)
cluster_0_summary_statistics = cluster_0_contents.describe()
cluster_0_summary_statistics.to_csv('spotify-2023-cluster_0_summary-statistics.csv', header=True)
print("Cluster 0 Contents:")
print(df[labels ==0])
print("Cluster 0 Summary Statistics:")
print(cluster_0_summary_statistics)

# Saving and printing cluster 1 contents
cluster_1_contents = df[labels ==1]
cluster_1_contents.to_csv('spotify-2023-cluster_1_contents.csv', header=True)
cluster_1_summary_statistics = cluster_1_contents.describe()
cluster_1_summary_statistics.to_csv('spotify-2023-cluster_1_summary-statistics.csv', header=True)
print("Cluster 1 Contents:")
print(df[labels ==1])
print("Cluster 1 Summary Statistics:")
print(cluster_1_summary_statistics)

# Saving and printing cluster 2 contents
cluster_2_contents = df[labels ==2]
cluster_2_contents.to_csv('spotify-2023-cluster_2_contents.csv', header=True)
cluster_2_summary_statistics = cluster_2_contents.describe()
cluster_2_summary_statistics.to_csv('spotify-2023-cluster_2_summary-statistics.csv', header=True)
print("Cluster 2 Contents:")
print(df[labels ==2])
print("Cluster 2 Summary Statistics:")
print(cluster_2_summary_statistics)





