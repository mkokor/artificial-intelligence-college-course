from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers


# LOADING DATASET
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()


# ANALYZING DATA
print(f"Training data dimensions: {xTrain.shape}")
print(f"Training labels dimensions: {yTrain.shape}")
print(f"Testing data dimensions: {xTest.shape}")
print(f"Testing labels dimensions: {yTest.shape}")

for i in range(1, 10):
  plt.subplot(3, 3, i)
  plt.imshow(xTrain[i])
  plt.axis("off")

plt.show()


# DATA PREPROCESSING
# One-hot encoding of labels!
# Data into type float and range [0, 1]!
yTrain = to_categorical(yTrain)
yTest = to_categorical(yTest)
xTrain = xTrain.astype("float") / 255
xTest = xTest.astype("float") / 255


# CREATING NEURAL NETWORK MODEL
def defineNeuralNetworkModel():
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), padding = "same", input_shape = (32, 32, 3)))
  model.add(layers.Conv2D(32, (3, 3), padding = "same"))
  model.add(layers.MaxPooling2D((2, 3)))
  model.add(layers.Dropout(0.2))
  model.add(layers.Conv2D(64, (3, 3), padding = "same"))
  model.add(layers.Conv2D(64, (3, 3), padding = "same"))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Dropout(0.2))
  model.add(layers.Conv2D(128, (3, 3), padding = "same"))
  model.add(layers.Conv2D(128, (3, 3), padding = "same"))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Dropout(0.2))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation = "relu"))
  model.add(layers.Dropout(0.2))
  model.add(layers.Dense(10, activation = "softmax"))
  customOptimizer = optimizers.SGD(learning_rate = 0.001, momentum = 0.9)
  model.compile(optimizer = customOptimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
  return model

model = defineNeuralNetworkModel()


# TRAINING
trainingHistory = model.fit(xTrain, yTrain, epochs = 30, batch_size = 64, validation_data = (xTest, yTest))
history = trainingHistory.history


# ANALYZING TRAINING RESULTS
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

accuracy = history["accuracy"]
validationAccuracy = history["val_accuracy"]
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, "bo", label = "Training accuracy")
plt.plot(epochs, validationAccuracy, "b", label = "Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


# EVALUATION
model.evaluate(xTest, yTest)