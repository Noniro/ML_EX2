import numpy as np
from sklearn.model_selection import train_test_split
from itertools import combinations


def load_versicolor_virginica(file_path):
    """Load only Versicolor and Virginica data from iris dataset"""
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5 or parts[4] == 'Iris-setosa':
                continue
            features = list(map(float, parts[1:3]))  # features 2 and 3
            label = 1 if parts[4] == 'Iris-versicolor' else -1
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)


def generate_hypotheses(X, y):
    """Generate all possible lines passing through pairs of points from different classes"""
    hypotheses = []
    for i, j in combinations(range(len(X)), 2):
        if y[i] != y[j]:  # points from different classes
            point1, point2 = X[i], X[j]

            # Calculate the weight vector (perpendicular to the line)
            w = point2 - point1
            # Rotate 90 degrees to get normal vector
            w = np.array([-w[1], w[0]])

            # Normalize only if norm is not too small
            norm = np.linalg.norm(w)
            if norm > 1e-10:  # Avoid division by very small numbers
                w = w / norm

                # Calculate bias term using midpoint
                midpoint = (point1 + point2) / 2
                b = -np.dot(w, midpoint)

                hypotheses.append((w, b))
    return hypotheses


def hypothesis_predict(weights, bias, X):
    """Predict using a single hypothesis (line)"""
    return np.sign(np.dot(X, weights) + bias)


def adaboost_train(X, y, hypotheses, n_classifiers=8):
    """Train AdaBoost classifier"""
    n_samples = len(y)
    weights = np.ones(n_samples) / n_samples

    selected_classifiers = []
    alphas = []

    for t in range(n_classifiers):
        min_error = float('inf')
        best_hypothesis = None
        best_predictions = None

        # Try each hypothesis
        for w, b in hypotheses:
            predictions = hypothesis_predict(w, b, X)
            error = np.sum(weights * (predictions != y))

            # Try flipped hypothesis
            error_opposite = np.sum(weights * (-predictions != y))

            if error_opposite < error:
                error = error_opposite
                predictions = -predictions
                w, b = -w, -b

            if error < min_error:
                min_error = error
                best_hypothesis = (w, b)
                best_predictions = predictions


        # Calculate alpha
        eps = min_error
        alpha = 0.5 * np.log((1 - eps) / (eps + 1e-10))

        # Store classifier and its weight
        selected_classifiers.append(best_hypothesis)
        alphas.append(alpha)

        # Update weights
        weights *= np.exp(-alpha * y * best_predictions)
        weights /= np.sum(weights)

        # Debug information
        # train_error = compute_error(X, y, selected_classifiers, alphas, len(selected_classifiers))
        # print(f"Iteration {t + 1}: error = {min_error:.4f}, alpha = {alpha:.4f}, training error = {train_error:.4f}")

    return selected_classifiers, alphas


def ensemble_predict(X, classifiers, alphas, k):
    """Make predictions using the ensemble of k classifiers"""
    if k > len(classifiers):
        k = len(classifiers)

    predictions = np.zeros(len(X))
    for i in range(k):
        w, b = classifiers[i]
        predictions += alphas[i] * hypothesis_predict(w, b, X)
    return np.sign(predictions)


def compute_error(X, y, classifiers, alphas, k):
    """Compute error for the ensemble of k classifiers"""
    predictions = ensemble_predict(X, classifiers, alphas, k)
    return np.mean(predictions != y)


def run_experiment(file_path, n_runs=100, n_classifiers=8):
    """Run multiple experiments and average results"""
    train_errors = np.zeros(n_classifiers)
    test_errors = np.zeros(n_classifiers)
    X, y = load_versicolor_virginica(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42069)

    # Load and split data

    for run in range(n_runs):

        # Generate hypotheses from training data only
        hypotheses = generate_hypotheses(X_train, y_train)

        # Train AdaBoost
        classifiers, alphas = adaboost_train(X_train, y_train, hypotheses, n_classifiers)

        # Compute errors for each k
        for k in range(1, n_classifiers + 1):
            if k <= len(classifiers):
                train_errors[k - 1] += compute_error(X_train, y_train, classifiers, alphas, k)
                test_errors[k - 1] += compute_error(X_test, y_test, classifiers, alphas, k)
            else:
                # If we stopped early, use the last available classifier's error
                train_errors[k - 1] += train_errors[len(classifiers) - 1]
                test_errors[k - 1] += test_errors[len(classifiers) - 1]

    # Average errors over all runs
    train_errors /= n_runs
    test_errors /= n_runs

    return train_errors, test_errors


if __name__ == "__main__":
    file_path = "iris.txt"
    train_errors, test_errors = run_experiment(file_path)

    print("\nAverage results over 100 runs:")
    print("k\tTraining Error\tTest Error")
    print("-" * 40)
    for k in range(8):
        print(f"H{k + 1}\t{train_errors[k]:.4f}\t\t{test_errors[k]:.4f}")