# in the iris.txt:
# X - axis is the second feature
# Y - axis is the third feature

import numpy as np
from itertools import combinations

def calculate_max_margin_brute_force(data, labels):
    max_margin = 0
    best_weights = None
    best_bias = None

    # Generate all combinations of 2 points from different classes
    for i, j in combinations(range(len(data)), 2):
        if labels[i] == labels[j]:
            continue

        # Calculate the line (weights and bias) that separates the two points
        point1, point2 = data[i], data[j]
        weights = point2 - point1
        bias = -np.dot(weights, (point1 + point2) / 2)

        # Calculate the margin for all points
        margins = []
        for k in range(len(data)):
            margin = abs(np.dot(weights, data[k]) + bias) / np.linalg.norm(weights)
            margins.append(margin)

        # The margin is the minimum distance to the decision boundary
        margin = min(margins)

        # Update the maximum margin if necessary
        if margin > max_margin:
            max_margin = margin
            best_weights = weights
            best_bias = bias

    return max_margin, best_weights, best_bias

# Load the dataset for Setosa and Versicolor
def load_data_setosa_versicolor(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if parts[4] == 'Iris-virginica':
                continue
            features = list(map(float, parts[1:3]))  # Use only the second and third features
            label = 1 if parts[4] == 'Iris-setosa' else -1
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)

# Load the dataset for Setosa and Virginica
def load_data_setosa_virginica(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if parts[4] == 'Iris-versicolor':
                continue
            features = list(map(float, parts[1:3]))  # Use only the second and third features
            label = 1 if parts[4] == 'Iris-setosa' else -1
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)

# Perceptron algorithm
class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.mistakes = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)

                if y[idx] * y_predicted <= 0:
                    self.weights += self.learning_rate * y[idx] * x_i
                    self.bias += self.learning_rate * y[idx]
                    self.mistakes += 1

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

# Main function to run the Perceptron algorithm
if __name__ == "__main__":
    # Function to calculate the Perceptron margin
    def calculate_perceptron_margin(X, y, weights, bias):
        margins = []
        for i in range(len(X)):
            margin = (np.dot(weights, X[i]) + bias) / np.linalg.norm(weights)
            margins.append(margin)
        return min(margins)

    # Run Perceptron on Setosa and Versicolor
    X_sv, y_sv = load_data_setosa_versicolor('iris.txt')
    perceptron_sv = Perceptron(learning_rate=0.1, n_iters=1000)
    perceptron_sv.fit(X_sv, y_sv)
    predictions_sv = perceptron_sv.predict(X_sv)

    print("Setosa and Versicolor:")
    print("Final weights vector:", perceptron_sv.weights)
    print("Final bias:", perceptron_sv.bias)
    print("Number of mistakes:", perceptron_sv.mistakes)
    true_max_margin_sv, _, _ = calculate_max_margin_brute_force(X_sv, y_sv)
    perceptron_margin_sv = calculate_perceptron_margin(X_sv, y_sv, perceptron_sv.weights, perceptron_sv.bias)
    print("True maximum margin:", true_max_margin_sv)
    print("Perceptron margin:", perceptron_margin_sv)

    # Run Perceptron on Setosa and Virginica
    X_sv, y_sv = load_data_setosa_virginica('iris.txt')
    perceptron_sv = Perceptron(learning_rate=0.1, n_iters=1000)
    perceptron_sv.fit(X_sv, y_sv)
    predictions_sv = perceptron_sv.predict(X_sv)

    print("\nSetosa and Virginica:")
    print("Final weights vector:", perceptron_sv.weights)
    print("Final bias:", perceptron_sv.bias)
    print("Number of mistakes:", perceptron_sv.mistakes)
    true_max_margin_sv, _, _ = calculate_max_margin_brute_force(X_sv, y_sv)
    perceptron_margin_sv = calculate_perceptron_margin(X_sv, y_sv, perceptron_sv.weights, perceptron_sv.bias)
    print("True maximum margin:", true_max_margin_sv)
    print("Perceptron margin:", perceptron_margin_sv)