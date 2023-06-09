# Resolving square equation...


def getInput(valueName):
  while True:
    inputValue = input(f"Enter {valueName}: ")
    try:
      return float(inputValue)
    except:
      print("Input value is not a number! Try again.")

def getParameters():
  a = getInput("a")
  b = getInput("b")
  c = getInput("c")
  return [a, b, c]

def calculateDiscriminant(a, b, c):
  return b ** 2 - 4 * a * c

def resolveSquareEquation(a, b, c):
  discriminant = calculateDiscriminant(a, b, c)
  if discriminant < 0:
    print("Results are complex!")
  elif discriminant == 0:
    print(f"Result is {-b / (2 * a)}.")
  else:
    x1 = (-b + discriminant ** (1 / 2)) / (2 * a)
    x2 = (-b - discriminant ** (1 / 2)) / (2 * a)
    print(f"Results are: {x1} i {x2}.")

def main():
  parameters = getParameters()
  resolveSquareEquation(parameters[0], parameters[1], parameters[2])


main()