# in the iris.txt:
# X - axis is the second feature
# Y - axis is the third feature


import numpy as np
from itertools import combinations

def calculate_max_margin_brute_force(data, labels):
    max_margin = 0
    best_weights = None
    best_bias = None

    # for pairs of points
    for i, j in combinations(range(len(data)), 2):
        if labels[i] == labels[j]:
            continue

        # Calculating the line separating the two points
        point1, point2 = data[i], data[j]
        weights = point2 - point1
        bias = -np.dot(weights, (point1 + point2) / 2)

        # compute the margin for each point
        margins = []
        for k in range(len(data)):
            margin = abs(np.dot(weights, data[k]) + bias) / np.linalg.norm(weights)
            margins.append(margin)

        # find the minimum margin
        margin = min(margins)

        # update the maximum margin
        if margin > max_margin:
            max_margin = margin
            best_weights = weights
            best_bias = bias

    # for triplets of points
    for i, j, k in combinations(range(len(data)), 3):
        if labels[i] == labels[j] and labels[i] != labels[k]:
            # i and j are in the same class, k is in the other class
            point1, point2, point3 = data[i], data[j], data[k]

            # vector normal to the line passing through point1 and point2
            direction = point2 - point1
            normal = np.array([-direction[1], direction[0]])  # perpendicular to the line

            # compute the bias
            mid_point = (point1 + point2) / 2
            bias = -np.dot(normal, mid_point)
            # check if the line is valid
            norm_normal = np.linalg.norm(normal)
            if norm_normal == 0:  # check if the line is vertical
                continue

            # The distance of all three points from the line
            margin1 = abs(np.dot(normal, point1) + bias) / np.linalg.norm(normal)
            margin2 = abs(np.dot(normal, point2) + bias) / np.linalg.norm(normal)
            margin3 = abs(np.dot(normal, point3) + bias) / np.linalg.norm(normal)
            margin = min(margin1, margin2, margin3)

            # update the maximum margin
            if margin > max_margin:
                max_margin = margin
                best_weights = normal
                best_bias = bias

    return max_margin, best_weights, best_bias


# Load the dataset for two classes
def load_data_setosa_versicolor(file_path, class1, class2):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if parts[4] == class1:
                continue
            features = list(map(float, parts[1:3]))  # Use only the second and third features
            label = 1 if parts[4] == class2 else -1
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)


def load_data_from_2_classes(file_path, class1, class2):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if parts[4] not in [class1, class2]:
                continue
            features = list(map(float, parts[1:3]))  # Use only the second and third features
            label = 1 if parts[4] == class1 else -1
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
            margin = y[i] * (np.dot(weights, X[i]) + bias) / np.linalg.norm(weights)
            margins.append(margin)
        return min(margins)


    # Run Perceptron on Setosa and Versicolor
    X_sv, y_sv = load_data_from_2_classes('iris.txt', 'Iris-setosa', 'Iris-versicolor')
    perceptron_sv = Perceptron(learning_rate=0.1, n_iters=1000)
    perceptron_sv.fit(X_sv, y_sv)
    predictions_sv = perceptron_sv.predict(X_sv)

    print("Setosa and Versicolor:")
    print("Final weights vector:", perceptron_sv.weights)
    print("Final bias:", perceptron_sv.bias)
    print("Number of mistakes:", perceptron_sv.mistakes)
    true_max_margin_sv, _,_, = calculate_max_margin_brute_force(X_sv, y_sv)
    perceptron_margin_sv = calculate_perceptron_margin(X_sv, y_sv, perceptron_sv.weights, perceptron_sv.bias)
    print("True maximum margin:", true_max_margin_sv)
    print("Perceptron margin:", perceptron_margin_sv)

    # Run Perceptron on Setosa and Virginica
    X_sv, y_sv = load_data_from_2_classes('iris.txt', 'Iris-setosa', 'Iris-virginica')
    perceptron_sv = Perceptron(learning_rate=0.1, n_iters=1000)
    perceptron_sv.fit(X_sv, y_sv)
    predictions_sv = perceptron_sv.predict(X_sv)

    print("\nSetosa and Virginica:")
    print("Final weights vector:", perceptron_sv.weights)
    print("Final bias:", perceptron_sv.bias)
    print("Number of mistakes:", perceptron_sv.mistakes)
    true_max_margin_sv, _,_, = calculate_max_margin_brute_force(X_sv, y_sv)
    perceptron_margin_sv = calculate_perceptron_margin(X_sv, y_sv, perceptron_sv.weights, perceptron_sv.bias)
    print("True maximum margin:", true_max_margin_sv)
    print("Perceptron margin:", perceptron_margin_sv)
