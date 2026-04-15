import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore

def load_data():
    data = pd.read_csv("houses.csv")
    x = data[["Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality"]].values
    y = data["House_Price"].values
    return x, y

def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std 

def fit(X, y):
    w = np.zeros(X.shape[1])  # one weight per feature
    b = 0
    learning_rate = 0.1
    epochs = 1000
    costs = []

    for epoch in range(epochs):
        y_pred = X @ w + b        # matrix multiplication
        error = y_pred - y

        dw = (1/len(y)) * X.T @ error
        db = (1/len(y)) * np.sum(error)

        w -= learning_rate * dw
        b -= learning_rate * db

        cost = (1/len(y)) * np.sum(error**2)
        costs.append(cost)

    return w, b, costs

def predict(X, w, b):
    return X @ w + b

def plot_costs(costs):
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title('Cost Function Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Cost (MSE)')
    plt.grid(True)
    plt.show()

x, y = load_data()
mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
x_normalized = normalize(x)
w, b, costs = fit(x_normalized, y)

plot_costs(costs)

# Accuracy
y_pred_all = predict(x_normalized, w, b)
rmse = np.sqrt((1/len(y)) * np.sum((y_pred_all - y)**2))
r2 = 1 - (np.sum((y - y_pred_all)**2) / np.sum((y - np.mean(y))**2))
print(f"RMSE: ${rmse:.2f}")
print(f"R²:   {r2:.4f}")

# Predict a single house
z = np.array([
    float(input("Square footage: ")),
    float(input("Num bedrooms: ")),
    float(input("Num bathrooms: ")),
    float(input("Year built: ")),
    float(input("Lot size: ")),
    float(input("Garage size: ")),
    float(input("Neighborhood quality: "))
])
z_normalized = (z - mean) / std
print(f"Predicted price: ${predict(z_normalized, w, b):.2f}")