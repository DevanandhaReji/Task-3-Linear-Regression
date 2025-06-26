# Task-3-Linear-Regression
Task 3: Linear Regression
 Objective
To implement and understand both simple and multiple linear regression using a housing dataset, and to evaluate model performance using metrics like MAE, MSE, and R² Score.
Tools Used
- Python
- Google Colab
- pandas
- scikit-learn
- matplotlib

Step 1: Import Required Libraries
pandas: To load and handle tabular data

numpy: (optional) for numerical operations

matplotlib.pyplot: For plotting graphs

sklearn.model_selection.train_test_split: To split dataset

sklearn.linear_model.LinearRegression: To create and train the model

sklearn.metrics: To evaluate model performance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

 Step 2:  Reading the Dataset
We upload the dataset (Housing.csv) into Google Colab and use pandas to read it.

Step 3: Preprocessing the Data
 3.1: Convert Categorical Variables to Numeric
Columns with values like "yes", "no", "furnished" are not usable as-is in machine learning.
We use pd.get_dummies() to convert them into binary (0/1) columns.

 3.2: Convert True/False to 1/0
If any columns have boolean values (True or False), they are also converted into integers.
Machine learning models can only process numerical data — all text/boolean values must be converted to numbers.

 Step 4: Selecting Features and Target
X (features): All the input columns used to predict the price (e.g., area, bedrooms, bathrooms, etc.)
y (target): The output column we want to predict (price)
To tell the model what inputs to use for learning and what output to predict.

 Step 5: Splitting the Dataset
We divide the dataset into:
Training set (80%): Used to teach the model.
Testing set (20%): Used to check how well the model performs on new data.
This is done using train_test_split().
To evaluate the model fairly on unseen data and avoid overfitting.

 Step 6: Training the Linear Regression Model
We create an instance of LinearRegression() and train it using the training set (X_train, y_train).
To allow the model to learn the relationship between features and price using mathematical regression.

 Step 7: Making Predictions
We use the trained model to predict prices on the testing set (X_test).
To see how well the model can predict prices for houses it hasn’t seen before.

 Step 8: Evaluating the Model
We evaluate model accuracy using three metrics:
MAE (Mean Absolute Error): Average of absolute errors.
MSE (Mean Squared Error): Average of squared errors.
R² Score (Coefficient of Determination): Indicates how well the model fits the data (closer to 1 is better).
To measure how close the predicted prices are to the actual prices.

Step 9: Interpreting Model Coefficients
We examine:
Intercept: The base price when all input features are zero.
Coefficients: The effect each feature has on the predicted price.
To understand the influence of each feature on the target variable.

Step 10 : Visualizing the Regression Line
If we use only one feature (e.g., area), we can plot a graph showing:
A scatter plot of actual values
A line showing predicted values
To visualize how well the regression line fits the data.

Step 11: Saving the Preprocessed File
After cleaning the data, we can save it as a new CSV file using to_csv() and download it from Colab.
To keep a copy of the cleaned data for reuse, analysis, or submission.

