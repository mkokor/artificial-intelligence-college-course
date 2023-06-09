# Finding n-th Fibonacci number...


def getNthFibonacciNumber(n):
  fibonacciArray = [0, 1, 1]
  if n <= len(fibonacciArray) - 1:
    return fibonacciArray[n]
  for i in range(3, n + 1):
    fibonacciArray.append(fibonacciArray[i - 2] + fibonacciArray[i - 1])
  return fibonacciArray[-1]

def getInput():
  while True:
    inputValue = input(f"Enter numeric value: ")
    try:
      n = int(inputValue)
      if n < 1:
        print("Input must be positive integer number! Try again.")
        continue
      return n
    except:
      print("Input value is not an integer! Try again.")

def main():
  n = getInput()
  print(f"{n}. Fibonacci number is {getNthFibonacciNumber(n)}.")


main()