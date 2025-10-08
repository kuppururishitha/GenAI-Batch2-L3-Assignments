#Assignemnt statement:
#Can you build a multivariate linear regression model that can predict the product sales based on the advertising budget allocated to different channels. The features are TV Budget ($),Radio Budget ($),Newspaper Budget ($) and the output is Sales (units)
#The dataset is give below
#TV Budget ($),Radio Budget ($),Newspaper Budget ($),Sales (units)
#230.1,37.8,69.2,22.1
#44.5,39.3,45.1,10.4
#17.2,45.9,69.3,9.3
#151.5,41.3,58.5,18.5
#180.8,10.8,58.4,12.9
#8.7,48.9,75.0,7.2
#57.5,32.8,23.5,11.8
#120.2,19.6,11.6,13.2
#144.1,16.0,40.3,15.6
#111.6,12.6,37.9,12.2

# step 1: import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# step 2: define the dataset 
data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 144.1, 111.6],
    'Radio': [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 16.0, 12.6],
    'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4, 75.0, 23.5, 11.6, 40.3, 37.9],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 15.6, 12.2]
}

# Convert to pandas DataFrame
dataset = pd.DataFrame(data)

# Input features
inputx = dataset[['TV', 'Radio', 'Newspaper']].values

# Output: Sales
outputy = dataset['Sales'].values


# step 3: split the data — 25% test, 75% training
input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size=1/4, random_state=0)

# step 4: train the Linear Regression model
model = LinearRegression()
model.fit(input_train, output_train)

# step 5: testing or model prediction using testinput
tv_budget = float(input("\nEnter the TV advertising budget ($): "))
radio_budget = float(input("Enter the Radio advertising budget ($): "))
newspaper_budget = float(input("Enter the Newspaper advertising budget ($): "))
testinput = [[tv_budget, radio_budget, newspaper_budget]]
predicted_output = model.predict(testinput)
print('\nThe test input (budgets) is:', testinput)
print('The predicted product sales (units) is:', predicted_output[0])
input("\nPress Enter to proceed...")

# step 6: print test data and model predictions
print("\nTest input data (TV, Radio, Newspaper budgets):\n", input_test)

# Predict on test set
predicted_output = model.predict(input_test)
print("\nPredicted Sales (units) on test set:\n", predicted_output)


