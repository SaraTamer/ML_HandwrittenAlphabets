import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import LinearSVC, SVC
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Data exploration and preparation:
# =====================================================================

# Load the dataset
file_path = "A_Z Handwritten Data.csv"  # Update the file path
data = pandas.read_csv(file_path)

# Work on a random subset of 5000 rows
data = data.sample(n=5000, random_state=42)

# Identify the number of unique classes
n_classes = data.loc[:, '0'].unique().size
print("Number of unique classes:", n_classes)

# show their distribution
# show their distribution
class_distribution = data.groupby('0').size()
print("\nClass Distribution")
print(class_distribution)

# Separate features columns and target column
X = data.drop(columns=['0'])
y = data['0']

# Normalize each image (each pixel is normalized to range from 0 to 1)
X = X / 255

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Reshape the flattened vectors to reconstruct and display the corresponding
# images while testing the models. [From requirement 1]

# each image is converted from [1D] 784 pixels into [2D] 28 x 28 pixels
X_test_reshaped = X_test.to_numpy().reshape(-1, 28, 28)

# First experiment:
# =====================================================================

# Train SVM model with linear kernel
linear_svm = LinearSVC(random_state=0)
linear_svm.fit(X_train, y_train)

# Test model on the testing subset
y_pred = linear_svm.predict(X_test)

# Confusion matrix size is 26 x 26 that corrspond to true labels and predicted labels
# Diagnal contains the correct classification made by the model [TP,TN]
# Other cells contain misclassifications
c_matrix = confusion_matrix(y_test, y_pred)

# confusion matrix visualization using heatmap

# class names
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z']

# configure heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(c_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=alphabet, yticklabels=alphabet)
plt.title('Confusion Matrix [SVM Linear Kernal]')
plt.ylabel('Actual labels')
plt.xlabel('Predicted labels')

# display heatmap
plt.show()

# average f1 score
# weighted f1 score is chosen because the imbalance classes
print(f"\nAverage F1 Score [SVM Linear Kernal]: {f1_score(y_test, y_pred, average='weighted'):.2f}")

# Train SVM model with non-linear kernel
nonlinear_svm = SVC(kernel='rbf')
nonlinear_svm.fit(X_train, y_train)

# Test model on the testing data
y_pred = nonlinear_svm.predict(X_test)

# confusion matrix for testing data
c_matrix = confusion_matrix(y_test, y_pred)

# confusion matrix visualization using heatmap

# configure heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(c_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=alphabet, yticklabels=alphabet)
plt.title('Confusion Matrix [SVM Non-Linear Kernal]')
plt.ylabel('Actual labels')
plt.xlabel('Predicted labels')

# display heatmap
plt.show()

# average f1 score
# weighted f1 score is chosen because the imbalance classes
print( f"\nAverage F1 Score [SVM Non-Linear Kernal]: {f1_score(y_test, y_pred, average='weighted'):.2f}")

# Second experiment:
# =====================================================================
#   o Split the training dataset into training and validation datasets.
#   o Second experiment (Build from scratch):
#     ▪ Implement logistic regression for one-versus-all multi-class
#       classification.
#     ▪ Train the model and plot the error and accuracy curves for the training
#       and validation data.
#     ▪ Test the model and provide the confusion matrix and the average f-1
#       scores for the testing dataset.

# Third experiment:
# =====================================================================
# o Third experiment (You can use TensorFlow):
#   ▪ Design 2 Neural Networks (with different number of hidden layers,
#     neurons, activations, etc.)
#   ▪ Train each one of these models and plot the error and accuracy curves
#     for the training data and validation datasets.
#   ▪ Save the best model in a separated file, then reload it.
#   ▪ Test the best model and provide the confusion matrix and the average
#     f-1 scores for the testing data.
#   ▪ Test the best model with images representing the alphabetical letters
#     for the names of each member of your team.
#  o Compare the results of the models and suggest the best model.




# Simple example of a neural network structure
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    Dense(26, activation='softmax') # Output layer (26 classes for A-Z)
])

# Preparing the model by specifying how it will learn and evaluate its performance
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Reshape X_train, X_test
X_train_reshaped = X_train.to_numpy().reshape(-1, 28, 28)


# Train the model
history = model.fit(X_train_reshaped, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Saving the model
model.save("model.h5")


# Reload and test the best model
model = tf.keras.models.load_model("model.h5")
test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=2)
print(f"Best Model Test Accuracy: {test_acc:.4f}")

