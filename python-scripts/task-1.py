# BINARY CLASSIFICATION OF SPAM AND HAM MESSAGES


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
import matplotlib.pyplot as plt


# LOADING DATA
# Adjust file path!
data = pd.read_table("/data/spam-detection-data.txt", sep = ",")
print(data.head(3))


# SPLITTING DATASET INTO FEATURES AND LABELS
messages = data["Message"]
labels = data["Label"]


# REMOVING NOISE (HTML TAGS) FROM DATA
def removeHtmlTags(text):
  result = text.split(">")[1].split("<")[0]
  return result

def removeNoise(messages):
  temporaryMessages = messages
  for i in range(0, temporaryMessages.shape[0]):
    temporaryMessages[i] = removeHtmlTags(temporaryMessages[i])
  return temporaryMessages

messages = removeNoise(messages)


# CHECKING DATA
print(f"FIRST SENTENCE LENGTH: {len(messages[0])}")
print(f"SECOND SENTENCE LENGHT: {len(messages[1])}")


# SPLITTING DATASET INTO TRAINING AND TESTING SETS
xTrain, xTest, yTrain, yTest = train_test_split(messages, labels, test_size = 0.1, random_state = 1)


# TOKENIZING TEXT
tokenizer = Tokenizer()
tokenizer.fit_on_texts(xTrain)
print(f"WORDS DICTIONARY: {tokenizer.word_index}")
xTrain = tokenizer.texts_to_sequences(xTrain)
xTest = tokenizer.texts_to_sequences(xTest)


# VECTORIZING TEXT
def vectorizeSequences(sequences, dimension = 4000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  return results

def vectorizeLabels(labels):
  results = np.zeros((len(labels)))
  for i, label in enumerate(labels):
    if label == "Spam":
      results[i] = 1
  return results

xTrain = vectorizeSequences(xTrain)
xTest = vectorizeSequences(xTest)
yTrain = vectorizeLabels(yTrain)
yTest = vectorizeLabels(yTest)


# CREATING NEURAL NETWORK MODEL
model = models.Sequential()
model.add(layers.Dense(8, activation = "relu", input_shape = (4000,)))
model.add(layers.Dense(8, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))


# COMPILING NEURAL NETWORK MODEL
model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy"])


# NEURAL NETWORK MODEL TRAINING
trainingHistory = model.fit(xTrain, yTrain, epochs = 5, batch_size = 128, validation_split = 0.3)


# ANALYZING 
lossValues = trainingHistory.history["loss"]
validationLossValues = trainingHistory.history["val_loss"]
accuracy = trainingHistory.history["accuracy"]
validationAccuracy = trainingHistory.history["val_accuracy"]
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


# EVALUATING
model.evaluate(xTest, yTest)


# REAL DATA TEST
message = "This is spam message!"
tokenizedMessage = tokenizer.texts_to_sequences([message])
vectorizedMessage = vectorizeSequences(tokenizedMessage)
prediction = model.predict(vectorizedMessage)
prediction