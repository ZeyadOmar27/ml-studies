import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def load_data():
    data = np.loadtxt("houses.csv", delimiter=",", skiprows=1) # Both x and y are NumPy arrays because np.loadtxt() automatically loads the data into NumPy arrays.
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def normalize(x):
    mean_x = np.mean(x)
    std_x = np.std(x)
    x_normalized = (x - mean_x) / std_x
    return x_normalized

def fit(x, y):
    w = 0
    b = 0
    learning_rate = 0.1
    epochs = 1000
    costs = []

    for epoch in range(epochs):
        y_pred = w * x + b  # Predicted value   
        error = y_pred - y  # Error

        dw = (1/len(x)) * np.sum(error * x)
        db = (1/len(x)) * np.sum(error)
            
        w -= learning_rate * dw
        b -= learning_rate * db
        cost = (1/len(x)) * np.sum(error**2)
        costs.append(cost)
    return w, b, costs

def predict(x, w, b):
    pred_y = (w * x) + b
    return pred_y

def plot_costs(costs):
        plt.figure(figsize=(10, 6))
        plt.plot(costs)
        plt.title('Cost Function Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Cost (MSE)')
        plt.grid(True)
        plt.show()

def plot_regression_line(x_normalized, x_original, y, w, b, mean_x, std_x):
    plt.figure(figsize=(10, 6)) 
    # Scatter plot of actual data
    plt.scatter(x_original, y, color="blue", label="Actual Data")

    # Regression line
    x_range = np.linspace(min(x_normalized), max(x_normalized), 100) # Normalized x values
    y_range = w * x_range + b
    x_range_original = (x_range * std_x) + mean_x 

    plt.plot(x_range_original, y_range, color="red", label="Regression Line")

    # Labels and title
    plt.xlabel("Square Feet")
    plt.ylabel("House Price")
    plt.title("House Prices vs. Square Feet")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to run the program
x, y = load_data()
x_normalized = normalize(x) # Normalize the x values
w, b, costs = fit(x_normalized, y) # Fit the model


z = float(input("House size in square feet: "))  
mean_x = np.mean(x)  
std_x = np.std(x)  
z_normalized = (z - mean_x) / std_x  
y_pred = predict(z_normalized, w, b)
print(f"Predicted house price: ${y_pred:.2f}")

plot_regression_line(x_normalized, x, y, w, b, mean_x, std_x)
plot_costs(costs)

#Accuracy measures
y_pred_all = predict(x_normalized, w, b)# Predict prices for all houses

mse = (1/len(y)) * np.sum((y_pred_all - y)**2) # Average squared error
rmse = np.sqrt(mse) # Root mean squared error

ss_res = np.sum((y - y_pred_all)**2) # Total squared error of the model
ss_tot = np.sum((y - np.mean(y))**2) # Total squared error of the mean
r2 = 1 - (ss_res / ss_tot)

print(f"RMSE: ${rmse:.2f}") # Average prediction error
print(f"R²:   {r2:.4f}") # Model accuracy



