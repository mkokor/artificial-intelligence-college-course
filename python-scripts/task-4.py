# Deep learning using Keras (MNIST dataset)...


# Importing MNIST dataset...
from keras.datasets import mnist                                                                                                                                       
from keras import models
from keras import layers
from keras.utils import to_categorical


# Preparing training and testing sets...
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()                                                                                                           

print(f"Training set dimensions: {xTrain.shape}")
print(f"Training set number of elements: {xTrain.shape[0]}")
print(f"Training set size (bytes): {xTrain.itemsize * xTrain.size}")

print(f"Testing set dimensions: {xTest.shape}")
print(f"Testing set number of elements: {xTest.shape[0]}")
print(f"Testing set size (bytes): {xTest.itemsize * xTest.size}")

# Preparing data for artificial neural network...
trainImages = xTrain.reshape((xTrain.shape[0], 28 * 28))                                                                                                
trainImages = trainImages.astype("float32") / 255

testImages = xTest.reshape((xTest.shape[0], 28 * 28))              
testImages = testImages.astype("float32") / 255

print(f"Training set size (bytes): {trainImages.itemsize * trainImages.size}")
print(f"Testing set size (bytes): {testImages.itemsize * testImages.size}")

# Creating neural network model...
neuralNetworkModel = models.Sequential()                                                                                                                    
neuralNetworkModel.add(layers.Dense(512, activation = "relu", input_shape = (28 * 28,)))                
neuralNetworkModel.add(layers.Dense(10, activation = "softmax"))

# Compiling model...
neuralNetworkModel.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics =["accuracy"])     

# Preparing labels...  
trainLabels = to_categorical(yTrain)                                                                                                                                                
testLabels = to_categorical(yTest)

# Neural network model training...
neuralNetworkModel.fit(trainImages, trainLabels, epochs = 5, batch_size = 128)                                                        

# Evaulating model...
testLoss, testAccuracy = neuralNetworkModel.evaluate(testImages, testLabels)                                                        
print(f"Accuracy (on testing data): {testAccuracy}")