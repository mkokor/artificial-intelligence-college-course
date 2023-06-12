# MULTICLASS CLASSIFICATION OF STACKOVERFLOW QUESTIONS


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
import matplotlib.pyplot as plt


# LOADING DATA
data = pd.read_csv("/data/stackoverflow-questions.csv")
print(f"Last three rows: {data.tail(3)}")


# SPLITTING DATASET INTO FEATURES AND LABELS
questions = data["post"]
labels = data["tags"]
print(f"Different programming languages: {set(labels)}")


# ENCODING LABELS (one-hot)
labelEncoder = LabelEncoder()
numericalLabels = labelEncoder.fit_transform(labels)
encodedLabels = to_categorical(numericalLabels)
print(f"Encoded labels: {encodedLabels}")


# SPLITTING DATASET INTO TRAINING AND TESTING SETS
xTrain, xTest, yTrain, yTest = train_test_split(questions, encodedLabels, test_size = 0.1)


# TOKENIZING DATA
tokenizer = Tokenizer(num_words = 500)
tokenizer.fit_on_texts(xTrain)
xTrain = tokenizer.texts_to_sequences(xTrain)
xTest = tokenizer.texts_to_sequences(xTest)


# VECTORIZING DATA
def vectorizeSequences(sequences, dimension = 500):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  return results

xTrain = vectorizeSequences(xTrain)
xTest = vectorizeSequences(xTest)


# CREATING NEURAL NETWORK MODEL
model = models.Sequential()
model.add(layers.Dense(32, activation = "relu", input_shape = (500,)))
model.add(layers.Dense(8, activation = "relu"))
model.add(layers.Dense(4, activation = "softmax"))


# COMPILING NEURAL NETWORK MODEL
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.summary()


# TRAINING
trainingHistory = model.fit(xTrain, yTrain, epochs = 8, batch_size = 8, validation_split = 0.25)
history = trainingHistory.history


# ANALYZING
accuracy = history["accuracy"]
validationAccuracy = history["val_accuracy"]
lossValues = history["loss"]
validationLossValues = history["val_loss"]
epochs = range(1, len(lossValues) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs, lossValues, "bo", label = "Training Loss")
plt.plot(epochs, validationLossValues, "b", label = "Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, "bo", label = "Training Accuracy")
plt.plot(epochs, validationAccuracy, "b", label = "Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.show()


# EVALUATION
model.evaluate(xTest, yTest)