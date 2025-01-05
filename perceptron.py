
import math

import numpy as np
from itertools import combinations


# in the iris.txt:
# X - axis is the second feature
# Y - axis is the third feature
from sklearn import svm

from itertools import combinations
import numpy as np
import math


def calculate_max_margin_brute_force3(x: np.ndarray, y: np.ndarray) -> (float, tuple):
    """
    find the true maximum margin (not estimated) between the two classes.
    we will find this with brute force:
    iterate over all sets of 3 points - that points will define a line that can separate the two classes.
    check if the current hyperplane is valid (i.e. all points are on the correct side of the hyperplane).
    calculate the margin for the current hyperplane.
    keep track of the maximum margin found, and return it.

    :param x: Feature.
    :param y: Labels.
    :return: Maximum margin and best_weights (slope and intercept).
    """
    max_margin = 0
    best_weights = (0, 0)  # (slope, intercept)

    for i, j, k in combinations(range(len(x)), 3):
        if y[i] == y[j] == y[k]:
            continue

        a, b, c = x[i], x[j], x[k]

        if y[i] == y[j]:
            p_1, p_2, p_3 = a, b, c
        elif y[i] == y[k]:
            p_1, p_2, p_3 = a, c, b
        else:
            p_1, p_2, p_3 = b, c, a

        if p_1[0] == p_2[0]:
            slope = 0
        else:
            slope = (p_1[1] - p_2[1]) / (p_1[0] - p_2[0])

        dist_p1_p3 = math.sqrt((p_1[0] - p_3[0]) ** 2 + (p_1[1] - p_3[1]) ** 2)
        dist_p2_p3 = math.sqrt((p_2[0] - p_3[0]) ** 2 + (p_2[1] - p_3[1]) ** 2)
        middle = [(p_1[0] + p_3[0]) / 2, (p_1[1] + p_3[1]) / 2] if dist_p1_p3 < dist_p2_p3 else [(p_2[0] + p_3[0]) / 2, (p_2[1] + p_3[1]) / 2]

        line = [slope, middle[1] - slope * middle[0]]

        valid = True
        for p in range(len(x)):
            y_line = line[0] * x[p][0] + line[1]
            if y_line < x[p][1] and y[p] == 1:
                valid = False
                break
            if y_line > x[p][1] and y[p] == -1:
                valid = False
                break

        if not valid:
            continue

        margin = float(abs(line[0] * p_3[0] - p_3[1] + line[1]) / math.sqrt(line[0] ** 2 + 1))
        if margin > max_margin:
            max_margin = margin
            best_weights = (line[0], line[1])

    for i, j in combinations(range(len(x)), 2):
        if y[i] == y[j]:
            continue

        p1, p2 = x[i], x[j]
        middle = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

        valid = True
        for p in range(len(x)):
            if x[p][0] > middle[0] and y[p] == 1:
                valid = False
                break
            if x[p][0] < middle[0] and y[p] == -1:
                valid = False
                break

        if not valid:
            continue

        margin = abs(p2[0] - p1[0]) / 2
        if margin > max_margin:
            max_margin = margin
            best_weights = (0, middle[0])  # vertical line, slope = 0, intercept = x-coordinate of middle

    return max_margin, best_weights
def calculate_max_margin_brute_force2(data, labels):
    max_margin = 0
    best_weights = None

    # Convert labels if needed (assuming Setosa is 1, others are -1)
    y = np.where(labels == 'Setosa', 1, -1)

    # Try different angles for the hyperplane through origin
    for theta in np.linspace(0, 2 * np.pi, 1000):
        # Create unit weight vector
        w = np.array([np.cos(theta), np.sin(theta)])

        # Check classification and find minimum margin
        correct_classification = True
        min_margin = float('inf')

        for i in range(len(data)):
            # Calculate signed distance from point to hyperplane
            distance = np.dot(w, data[i]) * y[i]

            if distance <= 0:  # Misclassification
                correct_classification = False
                break

            min_margin = min(min_margin, distance)

        if correct_classification and min_margin > max_margin:
            max_margin = min_margin
            best_weights = w

    return max_margin, best_weights

def SVM(data, labels):
    # יוצרים מודל SVM עם Kernel לינארי
    model = svm.SVC(kernel='linear', C=1)

    # מתאים את המודל לנתונים
    model.fit(data, labels)

    # חישוב הווקטור של ה-weight
    weights = model.coef_[0]
    # חישוב ה-margin (ההפוך למינימום המרחק)
    margin = 1 / np.linalg.norm(weights)

    return margin, weights


def calculate_max_margin_brute_force(data, labels):
    max_margin = 0
    best_weights = None

    for i, j in combinations(range(len(data)), 2):
        if labels[i] != labels[j]:
            continue

        point1, point2 = data[i], data[j]
        weights = point2 - point1
        midpoint = (point1 + point2) / 2
        other_class_points = [k for k in range(len(data)) if labels[k] != labels[i]]

        for k in other_class_points:
            point3 = data[k]
            shifted_weights = midpoint - point3
            final_weights = weights + shifted_weights

            # Check classification
            correct_classification = True
            for m in range(len(data)):
                point = data[m]
                classification = np.dot(final_weights, point)
                if (labels[m] == labels[i] and classification <= 0) or \
                        (labels[m] != labels[i] and classification > 0):
                    correct_classification = False
                    break

            if correct_classification:
                # Calculate true geometric margin
                weight_norm = np.linalg.norm(final_weights)
                if weight_norm == 0:
                    continue

                distances = [np.abs(np.dot(final_weights, data[n])) / weight_norm
                             for n in range(len(data))]
                margin = np.min(distances)

                if margin > max_margin:
                    max_margin = margin
                    best_weights = final_weights

    return max_margin, best_weights


# Load the dataset for two classes
def load_data_setosa_versicolor(file_path,class1,class2):
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

# Load the dataset for Setosa and Virginica

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

            margin = y[i] * (np.dot(weights, X[i])) / np.linalg.norm(weights)
            margins.append(margin)
        return min(margins)


    # Run Perceptron on Setosa and Versicolor

    X_sv, y_sv = load_data_from_2_classes('iris.txt', 'Iris-setosa', 'Iris-versicolor')
    perceptron_sv = Perceptron(learning_rate=0.1, n_iters=1000)

    perceptron_sv.fit(X_sv, y_sv)
    predictions_sv = perceptron_sv.predict(X_sv)

    print("Setosa and Versicolor:")
    print("Final weights vector:", perceptron_sv.weights)
    print("Number of mistakes:", perceptron_sv.mistakes)
    true_max_margin_sv, _ = calculate_max_margin_brute_force(X_sv, y_sv)
    perceptron_margin_sv = calculate_perceptron_margin(X_sv, y_sv, perceptron_sv.weights)
    print("True maximum margin:", true_max_margin_sv)
    print("Perceptron margin:", perceptron_margin_sv)

    # Run Perceptron on Setosa and Virginica

    X_sv, y_sv = load_data_from_2_classes('iris.txt', 'Iris-setosa', 'Iris-virginica')
    perceptron_sv = Perceptron(learning_rate=0.1, n_iters=1000)

    perceptron_sv.fit(X_sv, y_sv)
    predictions_sv = perceptron_sv.predict(X_sv)

    print("\nSetosa and Virginica:")
    print("Final weights vector:", perceptron_sv.weights)
    print("Number of mistakes:", perceptron_sv.mistakes)
    true_max_margin_sv, _ = calculate_max_margin_brute_force(X_sv, y_sv)
    perceptron_margin_sv = calculate_perceptron_margin(X_sv, y_sv, perceptron_sv.weights)
    print("True maximum margin:", true_max_margin_sv)
    print("Perceptron margin:", perceptron_margin_sv)
