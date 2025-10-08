import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_data_from_github():
    
    url = ("https://raw.githubusercontent.com/kuppururishitha/GenAI-Batch2-L3-Assignments/"
           "main/data.csv")
    # Read into pandas
    df = pd.read_csv(url, header=None)
    # Assuming the CSV columns are: SquareFeet, Bedrooms, Price (in that order)
    df.columns = ['SquareFeet', 'Bedrooms', 'Price']
    return df

def train_model(df):
    X = df[['SquareFeet', 'Bedrooms']].values
    y = df['Price'].values
    model = LinearRegression()
    model.fit(X, y)
    return model

def plot_3d_surface(df, model):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot actual data points
    ax.scatter(df['SquareFeet'], df['Bedrooms'], df['Price'], color='red', label='Actual Data')
    
    # Create a mesh grid for the surface
    x_range = np.linspace(df['SquareFeet'].min(), df['SquareFeet'].max(), 20)
    y_range = np.linspace(df['Bedrooms'].min(), df['Bedrooms'].max(), 20)
    xx, yy = np.meshgrid(x_range, y_range)
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)
    
    # Plot the regression plane
    ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='viridis')
    
    ax.set_xlabel('Square Feet')
    ax.set_ylabel('Bedrooms')
    ax.set_zlabel('Price')
    ax.set_title('3D House Price Prediction Surface')
    ax.legend()
    plt.show()

def predict_from_input(model):
    print("\nEnter values to predict house price:")
    sq = float(input("Square Feet: "))
    bd = float(input("Number of Bedrooms: "))
    x_in = np.array([[sq, bd]])
    pred = model.predict(x_in)[0]
    print(f"Predicted house price: {pred:.2f}")

def main():
    df = load_data_from_github()
    model = train_model(df)
    
    # Optionally, plot the 3D surface
    plot_3d_surface(df, model)
    
    # Predict from input
    predict_from_input(model)

if __name__ == '__main__':
    main()
