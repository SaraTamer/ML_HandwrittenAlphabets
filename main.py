import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import LinearSVC, SVC
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten


from sklearn.metrics import confusion_matrix, classification_report, f1_score


# Data exploration and preparation:
# =====================================================================

# Load the dataset
file_path = "A_Z Handwritten Data.csv"  # Update the file path
data = pandas.read_csv(file_path)

# # Work on a random subset of 10000 rows
data = data.sample(n=10000, random_state=42)

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

# plot the class distribution
plt.bar(alphabet, class_distribution)
plt.title('Class Distribution')
plt.show()

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

# # Train SVM model with linear kernel
# linear_svm = LinearSVC(random_state=0)
# linear_svm.fit(X_train, y_train)

# # Test model on the testing subset
# y_pred = linear_svm.predict(X_test)

# # Confusion matrix size is 26 x 26 that corrspond to true labels and predicted labels
# # Diagnal contains the correct classification made by the model [TP,TN]
# # Other cells contain misclassifications
# c_matrix = confusion_matrix(y_test, y_pred)

# # Reconstruct some images with model predctions
# display_images(X_test_reshaped, y_test, y_pred, 'SVM Linear Kernal')

# # confusion matrix visualization using heatmap

# # configure heatmap
# plt.figure(figsize=(16, 12))
# sns.heatmap(c_matrix, annot=True, fmt="d", cmap="Blues",
#             xticklabels=alphabet, yticklabels=alphabet)
# plt.title('Confusion Matrix [SVM Linear Kernal]')
# plt.ylabel('Actual labels')
# plt.xlabel('Predicted labels')

# # display heatmap
# plt.show()

# # average f1 score
# # weighted f1 score is chosen because the imbalance classes
# print(f"\nAverage F1 Score [SVM Linear Kernal]: {f1_score(y_test, y_pred, average='weighted'):.2f}")

# # Train SVM model with non-linear kernel
# nonlinear_svm = SVC(kernel='rbf')
# nonlinear_svm.fit(X_train, y_train)

# # Test model on the testing data
# y_pred = nonlinear_svm.predict(X_test)

# # Reconstruct some images with model predctions
# display_images(X_test_reshaped, y_test, y_pred, 'SVM Nonlinear Kernal')

# # confusion matrix for testing data
# c_matrix = confusion_matrix(y_test, y_pred)

# # confusion matrix visualization using heatmap

# # configure heatmap
# plt.figure(figsize=(16, 12))
# sns.heatmap(c_matrix, annot=True, fmt="d", cmap="Blues",
#             xticklabels=alphabet, yticklabels=alphabet)
# plt.title('Confusion Matrix [SVM Non-Linear Kernal]')
# plt.ylabel('Actual labels')
# plt.xlabel('Predicted labels')

# # display heatmap
# plt.show()

# # average f1 score
# # weighted f1 score is chosen because the imbalance classes
# print( f"\nAverage F1 Score [SVM Non-Linear Kernal]: {f1_score(y_test, y_pred, average='weighted'):.2f}")

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


# # Split the data into training and validation datasets
# X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
# # Reshape X_train, X_test
# X_train_reshaped = X_train.to_numpy().reshape(-1, 28, 28)
# X_validation_reshaped = X_validation.to_numpy().reshape(-1, 28, 28)


# # Simple example of a neural network structure
# model1 = Sequential([
#     Flatten(input_shape=(28, 28)),
#     Dense(128, activation='relu'),  # Hidden layer with 128 neurons
#     Dense(26, activation='softmax') # Output layer (26 classes for A-Z)
# ])

# # Preparing the model by specifying how it will learn and evaluate its performance
# model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# history1 = model1.fit(X_train_reshaped, y_train,validation_data = (X_validation_reshaped,y_validation), epochs=10, batch_size=32)


# # Simple example of a neural network structure
# model2 = Sequential([
#     Flatten(input_shape=(28, 28)),
#     Dense(256, activation='relu'), 
#     Dense(128, activation='tanh'),
#     Dense(64, activation='sigmoid'),
#     Dense(26, activation='softmax') # Output layer (26 classes for A-Z)
# ])

# # Preparing the model by specifying how it will learn and evaluate its performance
# model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# history2 = model2.fit(X_train_reshaped, y_train,validation_data = (X_validation_reshaped,y_validation), epochs=10, batch_size=32)


# # Evaluate both models on the test dataset
# test_loss1, test_acc1 = model1.evaluate(X_test_reshaped, y_test, verbose=2)
# test_loss2, test_acc2 = model2.evaluate(X_test_reshaped, y_test, verbose=2)

# # Compare the test accuracies and save the best model
# if test_acc1 > test_acc2:
#     print(f"Model 1 is better with accuracy: {test_acc1:.4f}")
#     model1.save("best_model.h5")
# else:
#     print(f"Model 2 is better with accuracy: {test_acc2:.4f}")
#     model2.save("best_model.h5")



# # Reload the best model
# best_model = tf.keras.models.load_model("best_model.h5")

# # Evaluate the best model on the test dataset
# test_loss, test_acc = best_model.evaluate(X_test_reshaped, y_test, verbose=2)
# print(f"Best Model Test Accuracy: {test_acc:.4f}")

# # Predict the labels for the test dataset
# y_pred = best_model.predict(X_test_reshaped)
# y_pred_classes = np.argmax(y_pred, axis=1)


# # Generate the confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred_classes)
# print("Confusion Matrix:")
# print(conf_matrix)

# # Create labels for the plot (A-Z)
# labels = [chr(i) for i in range(65, 91)]  # ASCII values for A-Z


# # Plot confusion matrix
# plt.figure(figsize=(20,20))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=labels, 
#             yticklabels=labels)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()


# # Generate classification report
# report = classification_report(y_test, y_pred_classes, 
#                              target_names=labels, 
#                              output_dict=True)

# # Print detailed classification report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_classes, target_names=labels))

# # Calculate and print average F1 score
# avg_f1 = report['weighted avg']['f1-score']
# print(f"\nAverage F1 Score: {avg_f1:.4f}")


# def plot_curves(history, model_name):
#     # Accuracy curves
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.title(f'{model_name} Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     # Loss curves
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title(f'{model_name} Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()

#     plt.show()

# plot_curves(history1, "Model 1")
# plot_curves(history2, "Model 2")



# Function to split and reshape training data
def split_and_reshape(X_train, y_train):
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    X_train_reshaped = X_train.to_numpy().reshape(-1, 28, 28)
    X_validation_reshaped = X_validation.to_numpy().reshape(-1, 28, 28)
    return X_train_reshaped, X_validation_reshaped, y_train, y_validation

def build_model1():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model2():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(256, activation='relu'),
        Dense(128, activation='tanh'),
        Dense(64, activation='sigmoid'),
        Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_validation, y_validation, epochs=10, batch_size=32):
    return model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=epochs, batch_size=batch_size)

def evaluate_and_save_best_model(model1, model2, X_test, y_test):
    test_loss1, test_acc1 = model1.evaluate(X_test, y_test, verbose=2)
    test_loss2, test_acc2 = model2.evaluate(X_test, y_test, verbose=2)

    if test_acc1 > test_acc2:
        print(f"Model 1 is better with accuracy: {test_acc1:.4f}")
        model1.save("best_model.h5")
    else:
        print(f"Model 2 is better with accuracy: {test_acc2:.4f}")
        model2.save("best_model.h5")

def evaluate_best_model(X_test, y_test):
    best_model = tf.keras.models.load_model("best_model.h5")

    # Get the predections for X_test data
    y_pred = best_model.predict(X_test)
    
    # Get the class with the height propability for each row
    y_pred_classes = np.argmax(y_pred, axis=1)

    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    # ASCII values for A-Z
    labels = [chr(i) for i in range(65, 91)]

    # Plot confusion matrix
    plt.figure(figsize=(20, 20))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Generate classification report
    report = classification_report(y_test, y_pred_classes, target_names=labels, output_dict=True)

    # Calculate and print average F1 score
    avg_f1 = report['weighted avg']['f1-score']
    print(f"\nAverage F1 Score: {avg_f1:.4f}")

# Function to plot accuracy and loss curves
def plot_curves(history, model_name):
    plt.figure(figsize=(12, 4))

    # Accuracy curves
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

# # Main workflow
# X_train_reshaped, X_validation_reshaped, y_train, y_validation = split_and_reshape(X_train, y_train)

# model1 = build_model1()
# model2 = build_model2()

# history1 = train_model(model1, X_train_reshaped, y_train, X_validation_reshaped, y_validation)
# history2 = train_model(model2, X_train_reshaped, y_train, X_validation_reshaped, y_validation)

# X_test_reshaped = X_test.to_numpy().reshape(-1, 28, 28)
# evaluate_and_save_best_model(model1, model2, X_test_reshaped, y_test)
# evaluate_best_model(X_test_reshaped, y_test)

# plot_curves(history1, "Model 1")
# plot_curves(history2, "Model 2")




##################### Testing with our names characters #################

def preprocess_image(image_path):
    # Open image in grayscale mode
    image = Image.open(image_path).convert('L')

    # Resize image to 28 X 28
    image = image.resize((28, 28))

    # Scaling pixel values
    image_array = np.array(image) / 255.0  
    return image_array

def predict_letter(image_array, model):
    image_array = image_array.reshape(1, 28, 28)
    predictions = model.predict(image_array)
    # Get height predection letter
    predicted_class = np.argmax(predictions, axis=1)[0]
    print (np.argmax(predictions, axis=1))

    # Convert class index to ASCII letter
    return chr(predicted_class + 65)

def alphabetical_test():
    best_model = tf.keras.models.load_model("best_model.h5")

    team_names = ["Ahmed", "Sara", "Esraa", "Shefaa", "Ganna"]

    for name in team_names:
        predictions = []
        for letter in name:
            image_path = f"{letter.upper()}.png"
            image_array = preprocess_image(image_path)
            predicted_letter = predict_letter(image_array, best_model)
            predictions.append(predicted_letter)
        print(f"Actual Name: {name.upper()}, Predicted Name: {''.join(predictions)}")


alphabetical_test()