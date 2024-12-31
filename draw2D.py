import numpy as np
import matplotlib.pyplot as plt
from perceptron import load_data_setosa_versicolor, Perceptron, calculate_max_margin_brute_force, \
    load_data_setosa_virginica


def plot_decision_boundary(X, y, weights, bias, title):
    # Plot the data points
    for i, point in enumerate(X):
        if y[i] == 1:
            plt.scatter(point[0], point[1], color='blue', marker='o', label='Setosa' if i == 0 else "")
        else:
            plt.scatter(point[0], point[1], color='red', marker='x', label='Versicolor' if i == 0 else "")

    # Plot the decision boundary
    x_values = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    y_values = -(weights[0] * x_values + bias) / weights[1]
    plt.plot(x_values, y_values, label='Decision Boundary', color='green')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Example usage with Setosa and Versicolor
    X_sv, y_sv = load_data_setosa_versicolor('iris.txt')
    perceptron_sv = Perceptron(learning_rate=0.1, n_iters=1000)
    perceptron_sv.fit(X_sv, y_sv)

    # Plot Perceptron decision boundary
    plot_decision_boundary(X_sv, y_sv, perceptron_sv.weights, perceptron_sv.bias, "Perceptron Decision Boundary")

    # Calculate and plot true maximum margin decision boundary
    true_max_margin_sv, true_weights, true_bias = calculate_max_margin_brute_force(X_sv, y_sv)
    plot_decision_boundary(X_sv, y_sv, true_weights, true_bias, "True Maximum Margin Decision Boundary")

    #Example usage with Setosa and Virginica
    X_sv, y_sv = load_data_setosa_virginica('iris.txt')
    perceptron_sv = Perceptron(learning_rate=0.1, n_iters=1000)
    perceptron_sv.fit(X_sv, y_sv)

    # Plot Perceptron decision boundary
    plot_decision_boundary(X_sv, y_sv, perceptron_sv.weights, perceptron_sv.bias, "Perceptron Decision Boundary")

    # Calculate and plot true maximum margin decision boundary
    true_max_margin_sv, true_weights, true_bias = calculate_max_margin_brute_force(X_sv, y_sv)
    plot_decision_boundary(X_sv, y_sv, true_weights, true_bias, "True Maximum Margin Decision Boundary")

