import math

import numpy as np
from itertools import combinations


# in the iris.txt:
# X - axis is the second feature
# Y - axis is the third feature


def calculate_max_margin_brute_force(data, labels):
    max_margin = 0
    best_weights = None

    # נבצע חיפוש בין כל זוגות נקודות מאותו class
    for i, j in combinations(range(len(data)), 2):
        if labels[i] != labels[j]:
            continue  # נתמקד רק בנקודות מאותו class

        # הנקודות מאותו class
        point1, point2 = data[i], data[j]

        # וקטור שמחבר את שתי הנקודות
        weights = point2 - point1

        # נחשב את האמצע בין שתי הנקודות
        midpoint = (point1 + point2) / 2

        # רשימת הנקודות מהמחלקה השנייה
        other_class_points = [k for k in range(len(data)) if labels[k] != labels[i]]

        for k in other_class_points:
            point3 = data[k]

            # חישוב הווקטור שמחבר את האמצע לנקודה השלישית
            shifted_weights = midpoint - point3

            # הזזת הווקטור כך שיעבור דרך האמצע
            final_weights = weights + shifted_weights

            # בדיקה אם כל הנקודות מסווגות נכון
            correct_classification = True
            for m in range(len(data)):
                point = data[m]
                classification = np.dot(final_weights, point)

                # נקודות מה-class הראשון צריכות להיות חיוביות
                if labels[m] == labels[i] and classification <= 0:
                    correct_classification = False
                    break
                # נקודות מה-class השני צריכות להיות שליליות
                elif labels[m] != labels[i] and classification > 0:
                    correct_classification = False
                    break

            # חישוב השוליים
            if correct_classification:
                # חישוב ה-margin לפי המרחק בין ההיפר-מישור לנקודות
                margin = np.min(
                    [np.abs(np.dot(final_weights, data[n])) for n in range(len(data))]
                )

                # עדכון ה-margin המקסימלי
                if margin > max_margin:
                    max_margin = margin
                    best_weights = final_weights

    return max_margin, best_weights


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


class Perceptron:
    def __init__(self,
                 n_iters=1000):  # n_iters - It ensures that the algorithm won't get stuck in an infinite loop if the data isn't linearly separable.
        """
        Perceptron algorithm with learning rate and no bias term.
        """
        self.n_iters = n_iters
        self.weights = None
        self.mistakes = 0

    def fit(self, X, y):
        """
        Train the Perceptron model.
        X: Feature matrix (numpy array of shape [n_samples, n_features])
        y: Labels (numpy array of shape [n_samples], values must be +1 or -1)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.mistakes = 0

        for _ in range(self.n_iters):
            no_mistake = True
            # Iterate over all points
            for idx, x_i in enumerate(X):
                # Compute linear output: w_t · x_i (no bias term)
                linear_output = np.dot(x_i, self.weights)
                y_predicted = np.sign(linear_output)

                # On mistake, update weights
                if y[idx] * y_predicted <= 0:  # There is a difference between the predicted and the actual
                    self.weights += y[idx] * x_i
                    self.mistakes += 1
                    no_mistake = False
                    break  # Exit the round as per the algorithm

            # Exit algorithm if no mistakes in this iteration
            if no_mistake:
                print(f"Algorithm converged after {_ + 1} iterations.")
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        return np.sign(linear_output)


# Main function to run the Perceptron algorithm
if __name__ == "__main__":
    # Function to calculate the Perceptron margin
    def calculate_perceptron_margin(X, y, weights):
        margins = []
        for i in range(len(X)):
            margin = np.dot(weights, X[i]) / np.linalg.norm(weights)
            margins.append(margin)
        return min(margins)


    # Run Perceptron on Setosa and Versicolor
    X_sv, y_sv = load_data_setosa_versicolor('iris.txt')
    perceptron_sv = Perceptron(n_iters=1000)
    perceptron_sv.fit(X_sv, y_sv)
    predictions_sv = perceptron_sv.predict(X_sv)

    print("Setosa and Versicolor:")
    print("Final weights vector:", perceptron_sv.weights)
    print("Number of mistakes:", perceptron_sv.mistakes)
    true_max_margin_sv, _ = calculate_max_margin_brute_force(X_sv, y_sv)
    # perceptron_margin_sv = calculate_perceptron_margin(X_sv, y_sv, perceptron_sv.weights)
    print("True maximum margin:", true_max_margin_sv)
    # print("Perceptron margin:", perceptron_margin_sv)

    # Run Perceptron on Setosa and Virginica
    X_sv, y_sv = load_data_setosa_virginica('iris.txt')
    perceptron_sv = Perceptron(n_iters=1000)
    perceptron_sv.fit(X_sv, y_sv)
    predictions_sv = perceptron_sv.predict(X_sv)

    print("\nSetosa and Virginica:")
    print("Final weights vector:", perceptron_sv.weights)
    print("Number of mistakes:", perceptron_sv.mistakes)
    true_max_margin_sv, _ = calculate_max_margin_brute_force(X_sv, y_sv)
    # perceptron_margin_sv = calculate_perceptron_margin(X_sv, y_sv, perceptron_sv.weights)
    print("True maximum margin:", true_max_margin_sv)
    # print("Perceptron margin:", perceptron_margin_sv)
