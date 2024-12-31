# Python 3.8
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import combinations



def split_data_versicolor_virginica(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if parts[4] == 'Iris-setosa':
                continue
            features = list(map(float, parts[1:3]))  # Use only the second and third features
            label = 1 if parts[4] == 'Iris-versicolor' else -1
            data.append(features)
            labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    # Split the data into 50% training and 50% test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, random_state=42)

    return X_train, X_test, y_train, y_test

def generate_hypothesis_set(X_train, y_train):
    hypotheses = []
    for (i, j) in combinations(range(len(X_train)), 2):
        if y_train[i] == y_train[j]:
            continue
        point1, point2 = X_train[i], X_train[j]
        weights = point2 - point1
        bias = -np.dot(weights, (point1 + point2) / 2)
        hypotheses.append((weights, bias))
    return hypotheses

def hypothesis_predict(weights, bias, X):
    return np.sign(np.dot(X, weights) + bias)

def adaboost(X_train, y_train, hypotheses, n_classifiers=8):
    n_samples = len(y_train)
    weights = np.ones(n_samples) / n_samples
    classifiers = []
    alphas = []

    for t in range(n_classifiers):
        min_error = float('inf')
        best_classifier = None
        best_alpha = 0

        for (w, b) in hypotheses:
            predictions = hypothesis_predict(w, b, X_train)
            error = np.sum(weights * (predictions != y_train))

            if error < min_error:
                min_error = error
                best_classifier = (w, b)
                best_alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

        classifiers.append(best_classifier)
        alphas.append(best_alpha)

        predictions = hypothesis_predict(best_classifier[0], best_classifier[1], X_train)
        weights *= np.exp(-best_alpha * y_train * predictions)
        weights /= np.sum(weights)

        # Debug prints
        # print(f"Classifier {t+1}:")
        # print(f"  Weights: {weights}")
        # print(f"  Alpha: {best_alpha}")
        # print(f"  Error: {min_error}")
        # print(f"  Classifier: {best_classifier}")

    return classifiers, alphas

def compute_error(H, alphas, X, y):
    n_samples = len(y)
    Hk = np.zeros(n_samples)
    for i in range(len(H)):
        Hk += alphas[i] * hypothesis_predict(H[i][0], H[i][1], X)
    Hk = np.sign(Hk)
    return np.mean(Hk != y)

def run_experiment(file_path, n_runs=100, n_classifiers=8):
    avg_train_errors = np.zeros(n_classifiers)
    avg_test_errors = np.zeros(n_classifiers)

    for _ in range(n_runs):
        X_train, X_test, y_train, y_test = split_data_versicolor_virginica(file_path)
        hypotheses = generate_hypothesis_set(X_train, y_train)
        classifiers, alphas = adaboost(X_train, y_train, hypotheses, n_classifiers)

        for k in range(1, n_classifiers + 1):
            train_error = compute_error(classifiers[:k], alphas[:k], X_train, y_train)
            test_error = compute_error(classifiers[:k], alphas[:k], X_test, y_test)
            avg_train_errors[k-1] += train_error
            avg_test_errors[k-1] += test_error

    avg_train_errors /= n_runs
    avg_test_errors /= n_runs

    return avg_train_errors, avg_test_errors



if __name__ == "__main__":
    file_path = 'iris.txt'
    avg_train_errors, avg_test_errors = run_experiment(file_path)
    for k in range(1, 9):
        print(f"H{k} - Average training error: {avg_train_errors[k-1]}, Average test error: {avg_test_errors[k-1]}")