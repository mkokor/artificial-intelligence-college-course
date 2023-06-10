# Deep learning using Keras (MNIST dataset)...


from keras.datasets import mnist                                                                                                                                       # Importing MNIST dataset...
from keras import models
from keras import layers
from keras.utils import to_categorical


(xTrain, yTrain), (xTest, yTest) = mnist.load_data()                                                                                                           # Preparing training and testing sets...

print(f"Training set dimensions: {xTrain.shape}")
print(f"Training set number of elements: {xTrain.shape[0]}")
print(f"Training set size (bytes): {xTrain.itemsize * xTrain.size}")

print(f"Testing set dimensions: {xTest.shape}")
print(f"Testing set number of elements: {xTest.shape[0]}")
print(f"Testing set size (bytes): {xTest.itemsize * xTest.size}")

trainImages = xTrain.reshape((xTrain.shape[0], 28 * 28))                                                                                                # Preparing data for artificial neural network...
trainImages = trainImages.astype("float32") / 255

testImages = xTest.reshape((xTest.shape[0], 28 * 28))              
testImages = testImages.astype("float32") / 255

print(f"Training set size (bytes): {trainImages.itemsize * trainImages.size}")
print(f"Testing set size (bytes): {testImages.itemsize * testImages.size}")

neuralNetworkModel = models.Sequential()                                                                                                                    # Creating neural network model...
neuralNetworkModel.add(layers.Dense(512, activation = "relu", input_shape = (28 * 28,)))                
neuralNetworkModel.add(layers.Dense(10, activation = "softmax"))

neuralNetworkModel.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics =["accuracy"])     # Compiling model...

trainLabels = to_categorical(yTrain)                                                                                                                                 # Preparing labels...                  
testLabels = to_categorical(yTest)

neuralNetworkModel.fit(trainImages, trainLabels, epochs = 5, batch_size = 128)                                                        # Neural network model training...

testLoss, testAccuracy = neuralNetworkModel.evaluate(testImages, testLabels)                                                        # Evaulating model...
print(f"Accuracy (on testing data): {testAccuracy}")