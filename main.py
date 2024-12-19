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

from sklearn.metrics import confusion_matrix, classification_report, f1_score


# Data exploration and preparation:
# =====================================================================

# Load the dataset
file_path = "A_Z Handwritten Data.csv"  # Update the file path
data = pandas.read_csv(file_path)

# # Work on a random subset of 1000 rows
# data = data.sample(n=5000, random_state=42)

# class names
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z']

# Identify the number of unique classes
n_classes = data.loc[:, '0'].unique().size
print("\nNumber of unique classes:", n_classes)

# show their distribution
class_distribution = data.groupby('0').size()
print("\nClass Distribution:")
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

def display_images(images, actual_labels, pred_labels, title):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Actual label: [{alphabet[actual_labels.iloc[i]]}], Predicted label: [{alphabet[pred_labels[i]]}]', fontsize=6)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
  
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

# Reconstruct some images with model predctions
display_images(X_test_reshaped, y_test, y_pred, 'SVM Linear Kernal')

# confusion matrix visualization using heatmap

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

# Reconstruct some images with model predctions
display_images(X_test_reshaped, y_test, y_pred, 'SVM Nonlinear Kernal')

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


# Split the data into training and validation datasets
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
# Reshape X_train, X_test
X_train_reshaped = X_train.to_numpy().reshape(-1, 28, 28)
X_validation_reshaped = X_validation.to_numpy().reshape(-1, 28, 28)


# Simple example of a neural network structure
model1 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    Dense(26, activation='softmax') # Output layer (26 classes for A-Z)
])

# Preparing the model by specifying how it will learn and evaluate its performance
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history1 = model1.fit(X_train_reshaped, y_train,validation_data = (X_validation_reshaped,y_validation), epochs=10, batch_size=32)


# Simple example of a neural network structure
model2 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'), 
    Dense(128, activation='tanh'),
    Dense(64, activation='sigmoid'),
    Dense(26, activation='softmax') # Output layer (26 classes for A-Z)
])

# Preparing the model by specifying how it will learn and evaluate its performance
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history2 = model2.fit(X_train_reshaped, y_train,validation_data = (X_validation_reshaped,y_validation), epochs=10, batch_size=32)


# Evaluate both models on the test dataset
test_loss1, test_acc1 = model1.evaluate(X_test_reshaped, y_test, verbose=2)
test_loss2, test_acc2 = model2.evaluate(X_test_reshaped, y_test, verbose=2)

# Compare the test accuracies and save the best model
if test_acc1 > test_acc2:
    print(f"Model 1 is better with accuracy: {test_acc1:.4f}")
    model1.save("best_model.h5")
else:
    print(f"Model 2 is better with accuracy: {test_acc2:.4f}")
    model2.save("best_model.h5")


# Reload the best model

# Reload and test the best model1
model1 = tf.keras.models.load_model("best_model.h5")
test_loss1, test_acc1 = model1.evaluate(X_test_reshaped, y_test, verbose=2)
print(f"Best Model Test Accuracy: {test_acc1:.4f}")



def plot_curves(history, model_name):
    # Accuracy curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss curves
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_curves(history1, "Model 1")
plot_curves(history2, "Model 2")

