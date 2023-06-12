# Classification of Iris flower using Sklearn...


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# Loading dataset...
iris = load_iris()                                                                                 			 

x = iris.data
y = iris.target
featureNames = iris.feature_names
targetNames = iris.target_names

print(f"Feature names: {featureNames}")
print(f"Target names: {targetNames}")
print(f"Data examples: {x[:5]}")

# Preparing dataset for training and evaluation...
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 1)            	

# Preparing k-Nearest Neighbor cassifier...
knnClassifier = KNeighborsClassifier(n_neighbors = 3)                                               		

# Training...
knnClassifier.fit(xTrain, yTrain)                                                                   			

# Testing...
yPredicted = knnClassifier.predict(xTest)                                                           		
accuracy = metrics.accuracy_score(yTest, yPredicted)
print(f"Accuracy: {accuracy}") 