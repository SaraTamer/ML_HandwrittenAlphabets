import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


def logisticRegression(x_train, y_train, x_test, y_test):
    """
    Perform multi-class logistic regression using one-vs-all strategy.
    """
    iterations = 100
    alpha = 0.1
    weights = []
    biases = []
    predict_values = []

    unique_classes = np.unique(y_train)

    # Initialize overall metrics
    train_error_history_overall = np.zeros(iterations)
    train_accuracy_history_overall = np.zeros(iterations)
    validation_error_history_overall = np.zeros(iterations)
    validation_accuracy_history_overall = np.zeros(iterations)

    for i in unique_classes:
        y_train_binary = (y_train == i).astype(int)
        y_test_binary = (y_test == i).astype(int)

        # Train the model and get histories
        weight, bias, train_error, train_accuracy, val_error, val_accuracy = trainModel(
            x_train, y_train_binary, x_test, y_test_binary, iterations, alpha)

        weights.append(weight)
        biases.append(bias)

        # Accumulate metrics
        train_error_history_overall += np.array(train_error)
        train_accuracy_history_overall += np.array(train_accuracy)
        validation_error_history_overall += np.array(val_error)
        validation_accuracy_history_overall += np.array(val_accuracy)

    # Average the accumulated metrics
    n_classes = len(unique_classes)
    train_error_history_overall /= n_classes
    train_accuracy_history_overall /= n_classes
    validation_error_history_overall /= n_classes
    validation_accuracy_history_overall /= n_classes

    # Plot overall metrics
    plot_metrics(
        train_error_history_overall,
        validation_error_history_overall,
        iterations,
        "Overall Error Curve",
        "Error"
    )
    plot_metrics(
        train_accuracy_history_overall,
        validation_accuracy_history_overall,
        iterations,
        "Overall Accuracy Curve",
        "Accuracy"
    )

    # Predict values for each class
    for index, label in enumerate(unique_classes):
        value = predict(x_test, weights[index], biases[index])
        predict_values.append(value)

    predict_values = np.array(predict_values)
    predicted_classes = np.argmax(predict_values, axis=0)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Overall accuracy: {accuracy * 100:.2f}%")
    cm = confusion_matrix(y_test, predicted_classes)
    avg_f1 = f1_score(y_test, predicted_classes, average='weighted')
    print("\nConfusion Matrix:\n", cm)
    print(f"\nAverage F1-Score: {avg_f1:.2f}")
    return predicted_classes


def trainModel(x_train, y_train, x_test, y_test, iterations, alpha):
    """
    Train logistic regression model using gradient descent.
    """
    n_samples, n_features = x_train.shape
    weight = np.zeros(n_features)
    bias = 0

    train_error_history = []
    train_accuracy_history = []
    validation_error_history = []
    validation_accuracy_history = []

    for i in range(iterations):
        error_value, sigmoid = equations(x_train, y_train, weight, bias, n_samples)
        train_error_history.append(error_value)

        train_predictions = np.where(sigmoid >= 0.5, 1, 0)
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_accuracy_history.append(train_accuracy)

        difference = sigmoid - y_train
        dw_value = (1 / n_samples) * np.dot(x_train.T, difference)
        db_value = (1 / n_samples) * np.sum(difference)

        weight -= alpha * dw_value
        bias -= alpha * db_value

        val_error, val_sigmoid = equations(x_test, y_test, weight, bias, x_test.shape[0])
        validation_error_history.append(val_error)

        val_predictions = np.where(val_sigmoid >= 0.5, 1, 0)
        val_accuracy = accuracy_score(y_test, val_predictions)
        validation_accuracy_history.append(val_accuracy)

    return weight, bias, train_error_history, train_accuracy_history, validation_error_history, validation_accuracy_history


def plot_metrics(train_history, val_history, iterations, title, ylabel):
    plt.figure(figsize=(8, 5))
    plt.plot(range(iterations), train_history, label="Training")
    plt.plot(range(iterations), val_history, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def equations(x, y, w, b, n):
    sigmoid = 1 / (1 + np.exp(-(np.dot(x, w) + b)))
    error_value = (1 / n) * (
        -(np.dot(y.T, np.log(sigmoid + 1e-8)) + np.dot((1 - y).T, np.log(1 - sigmoid + 1e-8))))
    return error_value, sigmoid


def predict(x_test, weight, bias):
    sigmoid = 1 / (1 + np.exp(-(np.dot(x_test, weight) + bias)))
    return sigmoid


def main():
    file_path = "A_Z Handwritten Data.csv"
    data = pd.read_csv(file_path)

    data = data.sample(n=10000, random_state=42)

    n_classes = data.iloc[:, 0].nunique()
    print("Number of unique classes:", n_classes)

    X = data.drop(columns=['0']) / 255.0
    y = data['0']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logisticRegression(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
