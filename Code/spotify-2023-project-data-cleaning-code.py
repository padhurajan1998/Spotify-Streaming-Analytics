# -*- coding: utf-8 -*-
import pandas as pd


# Reading Spotify data. 
df = pd.read_csv(r'C:\Users\Chavez\Documents\School Content\2023 Fall\INSY 5377 - Web & Social Analytics\Data and Analysis\spotify-2023.csv', encoding='Latin-1')


# Cleaning the data
# 1. Identifying column data types and converting to more appropriate data types
df.info()

# Converting all non text or categorical columns into numeric
categorical_columns = ['track_name', 'artist(s)_name', 'key', 'mode']
for col in df.columns:
    if col not in categorical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# 2. Identifying columns with missing or odd values and replacing values that likely won't greatly alter results (with appropriate reasoning)
df.info()

# Filling in missing values in the 'key' column with Unknown
df['key'] = df['key'].fillna('Unknown')

# Filling in missing values in the 'in_deezer_playlists' column with 0 since the data author was able to receive full data for 'in_deezer_charts'.
df['in_deezer_playlists'] = df['in_deezer_playlists'].fillna(0)


# Removing and verifying row 576 (Love Grows (Where My Rosemary Goes)) since we do not have streaming data
print(df.iloc[572:576])
df.drop(574, axis=0, inplace=True)
print(df.iloc[572:576])

# Saving the contents of the data frame holding our cleaned data to a clean file
csv_file_name = 'spotify-2023-project-data-cleaned.csv'
df.to_csv(r'\Users\Chavez\Documents\School Content\2023 Fall\INSY 5377 - Web & Social Analytics\Data and Analysis\spotify-2023-project-data-clean.csv', index=False) # Index set to false to not write row numbers

print(df.info())
