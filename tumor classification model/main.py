import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Hyperparameters
epochs = 1000
alpha = 0.1

def load_data():
    data = np.loadtxt("patient_tumor_data.csv", delimiter=",", skiprows=1)
    x = data[:, 1:]
    y = data[:, 0]
    return x, y

def normalize(x):
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    x_normalized = (x - mean_x) / std_x
    return x_normalized, mean_x, std_x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_cost(y, predictions):
    m = len(y)
    cost = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

def fit(x, y):
    n_samples, n_features = x.shape
    w = np.zeros(n_features)
    b = 0
    costs = []

    for _ in range(epochs):
        y_linear_pred = np.dot(x, w) + b
        predictions = sigmoid(y_linear_pred)
        error = predictions - y

        dw = (1 / n_samples) * np.dot(x.T, error)
        db = (1 / n_samples) * np.sum(error)

        w -= alpha * dw
        b -= alpha * db
        cost = compute_cost(y, predictions)
        costs.append(cost)
    return w, b, costs

def predict(x, w, b):
    y_linear_pred = np.dot(x, w) + b
    predictions = sigmoid(y_linear_pred)
    return (predictions >= 0.5).astype(int)

def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return (correct / total) * 100

# Load and normalize
x, y = load_data()
x_normalized, mean_x, std_x = normalize(x)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

# Train
w, b, costs = fit(x_train, y_train)

# Evaluate
y_pred_train = predict(x_train, w, b)
y_pred_test = predict(x_test, w, b)

print(f"Train Accuracy: {accuracy(y_train, y_pred_train):.2f}%")
print(f"Test Accuracy:  {accuracy(y_test, y_pred_test):.2f}%")

# Cost curve
plt.plot(range(epochs), costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Over Iterations")
plt.show()

# Scatter plot
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', alpha=0.7)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Tumor Data (Red = Malignant, Blue = Benign)")
plt.show()

