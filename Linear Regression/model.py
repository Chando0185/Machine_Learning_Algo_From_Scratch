import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

def mse(y_true, y_pred):
    n_samples = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / n_samples
    return mse


class LinearRegression:
    def __init__(self, learning_rate = 0.01, iterations = 1000):

        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self,X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.iterations):
            y_prediction = np.dot(X, self.weights) + self.bias

            dw = (2 / n_samples) * np.dot(X.T, (y_prediction - y))
            db = (2 / n_samples) * np.sum(y_prediction - y)

            self.weights = self.weights  - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)
            loss = mse(y, y_prediction)
            self.losses.append(loss)
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression(learning_rate = 0.01, iterations=2000)
model.fit(X, y)
prediction =model.predict(X)
print(f"Prediction {prediction}")
print(prediction)
loss = model.losses
print(f"Loss: {loss}")

plt.figure(figsize = (8, 5))
plt.plot(model.losses, label = "MSE Loss")
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample real estate data
data = {
    'area': np.random.uniform(1000, 5000, 1000),
    'bedrooms': np.random.randint(1, 6, 1000),
    'age': np.random.uniform(0, 40, 1000),
    'price': np.random.uniform(100000, 1000000, 1000)
}
df = pd.DataFrame(data)

# Prepare features and target
X = df[['area', 'bedrooms', 'age']].values
y = df['price'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print(f"Prediction: {y_pred}")

loss = mse(y_test, y_pred)
print(f"MSE Loss: {loss:.4f}")
print(model.losses)
# Plotting the loss curve
plt.figure(figsize=(8, 5))
plt.plot(model.losses, label="MSE Loss")
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()






