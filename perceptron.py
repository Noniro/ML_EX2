# in the iris.txt:
# X - axis is the second feature
# Y - axis is the third feature

import numpy as np


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
    # Run Perceptron on Setosa and Versicolor
    X_sv, y_sv = load_data_setosa_versicolor('iris.txt')
    perceptron_sv = Perceptron(learning_rate=0.1, n_iters=1000)
    perceptron_sv.fit(X_sv, y_sv)
    predictions_sv = perceptron_sv.predict(X_sv)

    print("Setosa and Versicolor:")
    print("Final weights vector:", perceptron_sv.weights)
    print("Final bias:", perceptron_sv.bias)
    print("Number of mistakes:", perceptron_sv.mistakes)
    margin_sv = 1 / np.linalg.norm(perceptron_sv.weights)
    print("True maximum margin:", margin_sv)

    # Run Perceptron on Setosa and Virginica
    X_sv, y_sv = load_data_setosa_virginica('iris.txt')
    perceptron_sv = Perceptron(learning_rate=0.1, n_iters=1000)
    perceptron_sv.fit(X_sv, y_sv)
    predictions_sv = perceptron_sv.predict(X_sv)

    print("\nSetosa and Virginica:")
    print("Final weights vector:", perceptron_sv.weights)
    print("Final bias:", perceptron_sv.bias)
    print("Number of mistakes:", perceptron_sv.mistakes)
    margin_sv = 1 / np.linalg.norm(perceptron_sv.weights)
    print("True maximum margin:", margin_sv)