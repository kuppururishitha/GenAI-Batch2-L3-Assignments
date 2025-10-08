#can you predict the employee attrition in an organization based on the following features. The features and the dataset are given below. use a classification model with KNN algorithm
#Features:
#Age: Age of the employee (numerical).
#JobRole: The job role/position of the employee (categorical).
#MonthlyIncome: Employee's monthly salary (numerical).
#JobSatisfaction: A rating from 1 to 4 indicating the employee's satisfaction with the job (numerical).
#YearsAtCompany: Number of years the employee has been at the company (numerical).
#Attrition: Target label indicating whether the employee left the company (1 for attrition, 0 for no attrition)
#Age,JobRole,MonthlyIncome,JobSatisfaction,YearsAtCompany,Attrition
#29,Sales Executive,4800,3,4,1
#35,Research Scientist,6000,4,8,0
#40,Laboratory Technician,3400,2,6,0
#28,Sales Executive,4300,3,3,1
#45,Manager,11000,4,15,0
#25,Research Scientist,3500,1,2,1
#50,Manager,12000,4,20,0
#30,Sales Executive,5000,2,5,0
#37, Laboratory Technician,3100,2,9,0
#26, Research Scientist,4500,3,2,1

# Step 1: Import the required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Step 2: Define the dataset 
dataset = pd.DataFrame({
    'Age': [29, 35, 40, 28, 45, 25, 50, 30, 37, 26],
    'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Sales Executive',
                'Manager', 'Research Scientist', 'Manager', 'Sales Executive', 'Laboratory Technician', 'Research Scientist'],
    'MonthlyIncome': [4800, 6000, 3400, 4300, 11000, 3500, 12000, 5000, 3100, 4500],
    'JobSatisfaction': [3, 4, 2, 3, 4, 1, 4, 2, 2, 3],
    'YearsAtCompany': [4, 8, 6, 3, 15, 2, 20, 5, 9, 2],
    'Attrition': [1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
})

# Step 3: Preprocess the data — Encode the 'JobRole' categorical column using label encoder
le = LabelEncoder()
dataset['JobRole'] = le.fit_transform(dataset['JobRole'])

# Split input features and target output
inputx = dataset[['Age', 'JobRole', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany']].values
outputy = dataset['Attrition'].values

# Step 4: Split the data into training and testing sets (25% test size)
input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size=0.25, random_state=0)

# Step 5: Select the KNN classifier model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(input_train, output_train)

# Step 6: Test the model using user input
age = float(input("\nEnter the employee's age: "))
job_role = input("Enter the employee's job role: ")
monthly_income = float(input("Enter the employee's monthly income: "))
job_satisfaction = int(input("Enter the job satisfaction level (1-4): "))
years_at_company = int(input("Enter the number of years at the company: "))

# Encode the job role input using the label encoder
encoded_job_role = le.transform([job_role])[0]

testinput = [[age, encoded_job_role, monthly_income, job_satisfaction, years_at_company]]
predicted_output = model.predict(testinput)

print("\nThe test input is:", testinput)
print("The predicted attrition (1 = will leave, 0 = will stay):", predicted_output[0])
