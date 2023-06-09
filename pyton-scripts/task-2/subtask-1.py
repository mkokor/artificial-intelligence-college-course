# Finding second greatest number in list...


def getNumber():
  while True:
    inputValue = input(f"Enter numeric value: ")
    try:
      return float(inputValue)
    except:
      print("Input value is not a number! Try again.")

def getListOfNumbers():
  listOfNumbers = []
  for i in range(1, 11):
    listOfNumbers.append(getNumber())
  return listOfNumbers

def getNumbersSorted(numbers):
  copy = numbers[:]
  copy.sort()
  return copy

def main():
  listOfNumbers = getListOfNumbers()
  print(f"Second greatest number is {getNumbersSorted(listOfNumbers)[-2]}.")


main()