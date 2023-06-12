import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from sklearn.preprocessing import StandardScaler


# LOADING DATA FROM CSV FILES
# Ajust file paths!
redWines = pd.read_csv("/data/red-wine-quality.csv", sep = ";")
whiteWines = pd.read_csv("/data/white-wine-quality.csv", sep = ";")


# ADDING COLUMN WITH DATA LABELS
redWines["label"] = 1
whiteWines["label"] = 0

# CONCATENATING DATA INTO ONE DataFrame OBJECT
wines = pd.concat([whiteWines, redWines])

# ANALYSING DATASET
#wines.describe()
#wines.hist()

# SPLITTING DATA INTO CHARACTERISTICS AND LABELS
y = wines["label"]
x = wines.drop(columns = ["label"]) 

# SPLITTING DATASET INTO TRAINING AND TESTING SETS
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)

# STANDARDIZING DATA VALUES
standardScaler = StandardScaler().fit(xTrain)
xTrain = standardScaler.transform(xTrain)
xTest = standardScaler.transform(xTest)

# CREATING NEURAL NETWORK MODEL
model = models.Sequential()
model.add(layers.Dense(8, activation = "relu", input_shape = (12,)))
model.add(layers.Dense(8, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))

# COMPILING NEURAL NETWORK MODEL
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# NEURAL NETWORK MODEL TRAINING
trainingHistory = model.fit(xTrain, yTrain, epochs = 20, batch_size = 16)

# EVALUATING NEURAL NETWORK MODEL
evaluatingResults = model.evaluate(xTest, yTest)