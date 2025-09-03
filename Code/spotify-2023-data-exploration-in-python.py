# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import csv
import seaborn as sns

# Reading Spotify data. 
df = pd.read_csv(r'C:\Users\Chavez\Documents\School Content\2023 Fall\INSY 5377 - Web & Social Analytics\Data and Analysis\spotify-2023-project-data-clean.csv', encoding='Latin-1')

sns.set_style("white")

# Plotting 'streams' data in lightgreen histogram
ax = df.hist(column='streams', bins=7, color='lightgreen', edgecolor='darkgreen')
ax[0][0].xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax[0][0].set_title("Spotify Stream Count Distribution", fontsize=16)
plt.tight_layout()
plt.ylabel('Frequency', fontsize=16)
plt.xlabel('Stream Count (in billions)', fontsize=16)
ax[0][0].tick_params(axis='x', labelsize=12)
ax[0][0].tick_params(axis='y', labelsize=12)
#plt.xlabel('', fontsize=16)
plt.show()

# Plotting 'artist_count' data in lightgreen histogram
ax = df.hist(column=['artist_count'], bins=8, color='lightgreen', edgecolor='darkgreen')
ax[0][0].set_title("Arist Count Distribution", fontsize=16)
plt.tight_layout()
months=[1,2,3,4,5,6,7,8]
plt.xticks(months)
plt.ylabel('Frequency', fontsize=16)
plt.xlabel('Number of Artists', fontsize=16)
ax[0][0].tick_params(axis='x', labelsize=12)
ax[0][0].tick_params(axis='y', labelsize=12)
plt.show()

# Plotting 'realease_year'data in lightgreen histogram
plt.rcParams['figure.figsize'] = [6, 4]
ax = df.hist(column=['released_year'], bins=20, color='lightgreen', edgecolor='darkgreen')
ax[0][0].set_title("Release Year Distribution", fontsize=16)
plt.tight_layout()
decades=[1930,1935,1940,1945,1950,1955,1960,1965,1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020]
plt.xticks(decades)
plt.ylabel('Frequency', fontsize=16)
plt.xlabel('Release Years (5 year intervals)', fontsize=16)
ax[0][0].tick_params(axis='x', labelsize=12, rotation=45)
ax[0][0].tick_params(axis='y', labelsize=12)
plt.show()

# Plotting 'released_month' data in lightgreen histogram
ax = df.hist(column=['released_month'], bins=12, color='lightgreen', edgecolor='darkgreen')
ax[0][0].set_title("Release Month Distribution", fontsize=16)
plt.tight_layout()
months=[1,2,3,4,5,6,7,8,9,10,11,12]
plt.xticks(months)
plt.ylabel('Frequency', fontsize=16)
plt.xlabel('Release Months', fontsize=16)
ax[0][0].tick_params(axis='x', labelsize=12)
ax[0][0].tick_params(axis='y', labelsize=12)
plt.show()

# Plotting 'release_day' data in lightgreen histograms
plt.rcParams['figure.figsize'] = [8, 4]
ax = df.hist(column=['released_day'], bins=31, color='lightgreen', edgecolor='darkgreen')
ax[0][0].set_title("Release Day Distribution", fontsize=16)
plt.tight_layout()
plt.ylabel('Frequency', fontsize=16)
plt.xlabel('Release Days', fontsize=16)
months=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30, 31]
plt.xticks(months)
ax[0][0].tick_params(axis='x', labelsize=12)
ax[0][0].tick_params(axis='y', labelsize=12)
plt.show()



# Plotting platform data in slategrey historgrams
ax = df.hist(column=['in_spotify_playlists','in_spotify_charts','in_apple_playlists','in_apple_charts','in_deezer_playlists','in_deezer_charts','in_shazam_charts'], bins=10, figsize=(10,10), color='slategrey', edgecolor='black')

# Spotify Playlist Distribution subplot
ax[0][0].set_title("Spotify Playlist Distribution", fontsize=16)
ax[0][0].tick_params(axis='x', labelsize=12)
ax[0][0].tick_params(axis='y', labelsize=12)

# Spotify Charts Distirbution subplot
ax[0][1].set_title("Spotify Charts Distribution", fontsize=16)
ax[0][1].tick_params(axis='x', labelsize=12)
ax[0][1].tick_params(axis='y', labelsize=12)

# Apple Playlist Distribution subplot
ax[0][2].set_title("Apple Playlist Distribution", fontsize=16)
ax[0][2].tick_params(axis='x', labelsize=12)
ax[0][2].tick_params(axis='y', labelsize=12)

# Apple Charts Distribution subplot
ax[1][0].set_title("Apple Charts Distribution", fontsize=16)
ax[1][0].tick_params(axis='x', labelsize=12)
ax[1][0].tick_params(axis='y', labelsize=12)

# Deezer Playlist Distribution subplot
ax[1][1].set_title("Deezer Playlists Distribution", fontsize=16)
ax[1][1].tick_params(axis='x', labelsize=12)
ax[1][1].tick_params(axis='y', labelsize=12)

# Deezer Charts Distribution subplot
ax[1][2].set_title("Deezer Charts Distribution", fontsize=16)
ax[1][2].tick_params(axis='x', labelsize=12)
ax[1][2].tick_params(axis='y', labelsize=12)

# Shazam Charts Distribution subplot
ax[2][0].set_title("Shazam Charts Distribution", fontsize=16)
ax[2][0].tick_params(axis='x', labelsize=12)
ax[2][0].tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()

# Plotting song trait data in limegreen histograms
df.hist(column=['bpm','danceability_%','valence_%','energy_%','acousticness_%','instrumentalness_%','liveness_%','speechiness_%'], bins=10, figsize=(10,10),color='limegreen', edgecolor='darkgreen')
plt.tight_layout()
plt.show()



# Creates a green correlation heatmap matrix based on all numerical data
numeric_data = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap= 'Greens')
plt.title("Correlation Matrix", fontsize=40)
plt.tick_params(axis='x', rotation=45, bottom=True)
plt.show()


# Creates box and whisker plots for the numerical song traits except for bpm, mode, and key since they are not on percentage scale
plt.rcParams['figure.figsize'] = [8, 4]
song_trait_columns = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 
                   'liveness_%', 'speechiness_%']

spotify_colors = ['#1DB954', '#191414', '#B3B3B3']

boxprops = dict(facecolor=spotify_colors[0], color=spotify_colors[0], linewidth=2)
whiskerprops = dict(color=spotify_colors[2], linewidth=2)
capprops = dict(color=spotify_colors[1], linewidth=2)
medianprops = dict(color=spotify_colors[1], linewidth=2)
flierprops = dict(markerfacecolor=spotify_colors[0], marker='o', markersize=8, linestyle='none')
meanprops={"marker": "+","markeredgecolor": "black","markersize": "10"}

fig, ax = plt.subplots()
ax.boxplot(df[song_trait_columns].values, labels=song_trait_columns, boxprops=boxprops, whiskerprops=whiskerprops,
           capprops=capprops, medianprops=medianprops, flierprops=flierprops, meanprops=meanprops, showmeans=True, patch_artist=True, notch=True)
plt.title("Percentile Song Trait Summary Plot")
ax.set_ylabel('Percentage (%)')
plt.xticks(rotation=35)
plt.show(ax)

# Produces summary statistic numbers for the box&whisker plots and saves it to a CSV file in the current directory
# summary_statistics = df.describe()
# summary_statistics.to_csv('spotify-2023-summary statistics.csv', header=True)
plt.rcParams['figure.figsize'] = [4, 4]
song_trait_columns = ['bpm']

spotify_colors = ['#1DB954', '#191414', '#B3B3B3']

boxprops = dict(facecolor=spotify_colors[0], color=spotify_colors[0], linewidth=2)
whiskerprops = dict(color=spotify_colors[2], linewidth=2)
capprops = dict(color=spotify_colors[1], linewidth=2)
medianprops = dict(color=spotify_colors[1], linewidth=2)
flierprops = dict(markerfacecolor=spotify_colors[0], marker='o', markersize=8, linestyle='none')
meanprops={"marker": "+","markeredgecolor": "black","markersize": "10"}

fig, ax = plt.subplots()
ax.boxplot(df[song_trait_columns].values, labels=song_trait_columns, boxprops=boxprops, whiskerprops=whiskerprops,
           capprops=capprops, medianprops=medianprops, flierprops=flierprops, meanprops=meanprops, showmeans=True, patch_artist=True, notch=True)
plt.title("Beats Per Minute Summary Plot")
ax.set_ylabel('Number of Beats per Minute')
plt.show(ax)