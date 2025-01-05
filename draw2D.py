import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron, calculate_max_margin_brute_force, load_data_from_2_classes
from Q3_Adaboost import run_experiment, hypothesis_predict, load_versicolor_virginica, train_test_split, generate_hypotheses, adaboost_train


def plot_perceptron_decision_boundary(X, y, weights, title):
    # Plot the data points
    for i, point in enumerate(X):
        if y[i] == 1:
            plt.scatter(point[0], point[1], color='blue', marker='o', label='Setosa' if i == 0 else "")
        else:
            plt.scatter(point[0], point[1], color='red', marker='x', label='Versicolor' if i == 0 else "")

    # Plot the decision boundary
    x_values = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    y_values = -(weights[0] * x_values) / weights[1]
    plt.plot(x_values, y_values, label='Decision Boundary', color='green')

    plt.xlabel('Feature 2')
    plt.ylabel('Feature 3')
    plt.legend()
    plt.title(title)
    plt.show()


def plot_decision_boundary(classifiers, alphas, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = np.zeros(xx.shape)
    for i in range(len(classifiers)):
        Z += alphas[i] * hypothesis_predict(classifiers[i][0], classifiers[i][1],
                                            np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    Z = np.sign(Z)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    print("Press 1 for Perceptron and 2 for Adaboost")
    choice = input("Enter your choice: ")

    if choice == '1':
        # Example usage with Setosa and Versicolor

        X_sv, y_sv = load_data_from_2_classes('iris.txt', 'Iris-setosa', 'Iris-versicolor')
        perceptron_sv = Perceptron(learning_rate=0.1, n_iters=1000)

        perceptron_sv.fit(X_sv, y_sv)

        # Plot Perceptron decision boundary
        plot_perceptron_decision_boundary(X_sv, y_sv, perceptron_sv.weights, "Perceptron Decision Boundary")

        # Calculate and plot true maximum margin decision boundary
        true_max_margin_sv, true_weights = calculate_max_margin_brute_force(X_sv, y_sv)
        plot_perceptron_decision_boundary(X_sv, y_sv, true_weights, "True Maximum Margin Decision Boundary")

        # Example usage with Setosa and Virginica

        X_sv, y_sv = load_data_from_2_classes('iris.txt', 'Iris-setosa', 'Iris-virginica')
        perceptron_sv = Perceptron(learning_rate=0.1, n_iters=1000)

        perceptron_sv.fit(X_sv, y_sv)

        # Plot Perceptron decision boundary
        plot_perceptron_decision_boundary(X_sv, y_sv, perceptron_sv.weights, "Perceptron Decision Boundary")

        # Calculate and plot true maximum margin decision boundary

        true_max_margin_sv, true_weights, true_bias = calculate_max_margin_brute_force(X_sv, y_sv)
        plot_perceptron_decision_boundary(X_sv, y_sv, true_weights, true_bias, "True Maximum Margin Decision Boundary")


    elif choice == '2':

        file_path = 'iris.txt'

        train_errors, test_errors = run_experiment(file_path, n_runs=100, n_classifiers=8)

        # Plot average training and test errors

        plt.plot(range(1, 9), train_errors, label='Training Error')

        plt.plot(range(1, 9), test_errors, label='Test Error')

        plt.xlabel('Number of Classifiers')

        plt.ylabel('Error')

        plt.title('AdaBoost Training and Test Errors')

        plt.legend()

        plt.show()

        # Load data for plotting decision boundaries

        X, y = load_versicolor_virginica(file_path)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Generate hypotheses and train AdaBoost

        hypotheses = generate_hypotheses(X_train, y_train)

        classifiers, alphas = adaboost_train(X_train, y_train, hypotheses, n_classifiers=8)

        # Plot decision boundaries for H1 to H8

        for k in range(1, 9):

            plot_decision_boundary(classifiers[:k], alphas[:k], X_train, y_train, f'H{k} Decision Boundary')


        else:
            print("Invalid choice")

