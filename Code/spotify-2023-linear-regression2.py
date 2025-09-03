import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Reading Spotify data.
df = pd.read_csv(r'C:\Users\Chavez\Documents\School Content\2023 Fall\INSY 5377 - Web & Social Analytics\Data and Analysis\spotify-2023-project-data-clean.csv', encoding='Latin-1')

# List of columns to create linear regression models and graphs
columns_of_interest = ['artist_count','bpm','danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

for column in columns_of_interest:
    # Select specific columns for the features (X) and the target variable (y)
    X = df[[column]]
    y = df['streams']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Get the coefficients
    slope = model.coef_[0]
    intercept = model.intercept_

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    print(f'R-squared for {column}: {r2}')
    
    # Print the formula for the linear regression line
    formula = f'Linear Regression Line for {column}:\n y = {slope:.4f} * x + {intercept:.4f}'
    print(formula)

    # Visualize the results (for a single feature regression)
    plt.scatter(X_test[column], y_test, color='black')
    plt.plot(X_test[column], y_pred, color='blue', linewidth=3)
    plt.xlabel(column)
    plt.ylabel('Streams (in billions)')
    
    # Include the formula for the linear regression line
    formula = f'Linear Regression Line:\n y = {slope:.4f} * x + {intercept:.4f}'
    plt.text(0.1, 0.9, formula, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
    plt.title(f'Linear Regression Results - {column}')
    plt.ticklabel_format(axis='y', style='sci', useMathText=True) 
    plt.show()
