# Basic operations with matrices...


import numpy as np


zerosOnly = np.zeros((4, 3))
print(f"Matrix with zeros only: {zerosOnly}")

onesOnly = np.ones((2, 2))
print(f"Matrix with ones only: {onesOnly}")

vectorColumn = np.arange(1, 11, 1).reshape(10, 1)
print(f"Vector column example: {vectorColumn}")

matrix = np.array([1, 2, 3, 1 / 3.6, 5, 23, 2 ** 10.5, 42, np.cos(80.841)]).reshape(3, 3)
print(f"Matrix with specific numerical values: {matrix}")

transposeMatrix = np.transpose(matrix)
print(f"Transpose matrix: {transposeMatrix}")

sum = transposeMatrix + matrix
print(f"Sum of matrix and its transpose version: {sum}")

columnElementsSum = np.sum(matrix[:, 0])
product = matrix * columnElementsSum
print(f"Product of matrix and its first column sum: {product}")

thirdRowOfMatrix = matrix[2, :]
print(f"Third row of matrix: {thirdRowOfMatrix}")

secondColumnOfMatrix = matrix[:, 1]
print(f"Second column of matrix: {secondColumnOfMatrix}")

specificElement = matrix[0, 2]
print(f"Element located at (0, 2): {specificElement}")

def getSquareMatrix(oneDimensionalArray):
  validation = oneDimensionalArray.size ** (1 / 2) == float(oneDimensionalArray.size ** (1 / 2) // 1)
  if validation:
    return oneDimensionalArray.reshape((int(oneDimensionalArray.size ** (1 / 2)), int(oneDimensionalArray.size ** (1 / 2))))
  else: 
    raise Exception("Invalid parameter!")
    
try:
  print(getSquareMatrix(np.array([1, 2, 3, 4])))
except:
  print("An error occured!")