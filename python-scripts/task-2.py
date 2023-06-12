# Data normalization using Sklearn...


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Adjust file path!
data = pd.read_csv("../report.csv", na_values = "/")

# Creating imputers for handling NaN values...
simpleImputerMedian = SimpleImputer(strategy = "median")                                                                                               
simpleImputerMean = SimpleImputer(strategy = "mean")             

# Creating MinMax scaler...
minMaxScaler = MinMaxScaler()                                                                              				  

 # Replacing NaN values with "median" strategy...
data["Ispit1"] = simpleImputerMedian.fit_transform(data["Ispit1"].values.reshape(-1, 1))                    		 
data["Ispit2"] = simpleImputerMedian.fit_transform(data["Ispit2"].values.reshape(-1, 1))

 # Replacing NaN values with "mean" strategy...
data["Ispit1_popravni"] = simpleImputerMean.fit_transform(data["Ispit1_popravni"].values.reshape(-1, 1))   	 
data["Ispit2_popravni"] = simpleImputerMean.fit_transform(data["Ispit2_popravni"].values.reshape(-1, 1))

# Normalizing values using Z-score normalization...
data["Ispit1"] = scale(data["Ispit1"])                                                                      				  
data["Ispit2"] = scale(data["Ispit2"])

# Normalizing values using MinMax scaler...
data["Ispit1_popravni"] = minMaxScaler.fit_transform(data["Ispit1_popravni"].values.reshape(-1, 1))         		 
data["Ispit2_popravni"] = minMaxScaler.fit_transform(data["Ispit2_popravni"].values.reshape(-1, 1))       

# Other NaN values will be 0...
data.replace(np.nan, 0, inplace = True)                                                                     				 

# Deleting column with grades...
grades = data["Ocjena"]
data.drop(columns = ["Ocjena"], inplace = True)                                                             				  

# Converting DataFrame object to NumPy array...
gradesNumPyArray = grades.to_numpy()                                                                        				  
dataNumPyArray = data.to_numpy()

# Preparing for classification training...
xTrain, xTest, yTrain, yTest = train_test_split(dataNumPyArray, gradesNumPyArray, test_size = 0.2)          		  