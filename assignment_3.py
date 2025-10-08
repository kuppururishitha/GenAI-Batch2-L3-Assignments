#write a python program to draw the neural network for the the pima indians diabetes prediction problem which was discussed in the class
 
# Step 1: Import required libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 2: Load the Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
dataset = pd.read_csv(url, header=None, names=column_names)

# Step 3: Split into input features (X) and target (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Step 4: Split data into train and test sets with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Build the neural network model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Step 7: Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 8: Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
print("\n Model training completed.")

# Step 9: Evaluate the model on the test set
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n Test Accuracy: {accuracy:.4f}")

# Step 10: Make predictions and print classification report
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
print("\n Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Step 11: Take user input and predict
print("\n Enter the following details to predict diabetes:")

user_input = []
user_input.append(float(input("Pregnancies: ")))
user_input.append(float(input("Glucose: ")))
user_input.append(float(input("Blood Pressure: ")))
user_input.append(float(input("Skin Thickness: ")))
user_input.append(float(input("Insulin: ")))
user_input.append(float(input("BMI: ")))
user_input.append(float(input("Diabetes Pedigree Function: ")))
user_input.append(float(input("Age: ")))

# Convert to numpy array and reshape for scaler and model input
user_data = np.array(user_input).reshape(1, -1)
user_data_scaled = scaler.transform(user_data)

# Predict
prediction_prob = model.predict(user_data_scaled)[0][0]
prediction = 1 if prediction_prob > 0.5 else 0

print(f"\n Predicted probability of diabetes: {prediction_prob:.4f}")
print(f" Predicted class: {'Diabetic' if prediction == 1 else 'Not Diabetic'}")

# Step 12: Visualize the input compared to training data (PCA)
print("\n Visualizing where your input lies among the training data...")

# Combine input with training data for PCA
X_combined = np.vstack([X_train, user_data_scaled])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined)

# Split back into training and input after PCA
X_train_pca = X_pca[:-1]
user_input_pca = X_pca[-1]

# Plot
plt.figure(figsize=(8, 6))
colors = ['green' if label == 0 else 'red' for label in y_train]
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=colors, alpha=0.5, label='Training Data')

# Plot input
plt.scatter(user_input_pca[0], user_input_pca[1], c='blue', s=100, edgecolors='black', label='Your Input')

# Labels
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Visualization of Diabetes Prediction')
plt.legend()
plt.grid(True)
plt.show()
