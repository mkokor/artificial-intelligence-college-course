# Calculating binomial coefficient...


def factoriel(n):
  if n < 0:
    raise Exception("Invalid parameter!")
  elif n == 0 or n == 1:
    return 1
  else:
    return n * factoriel(n - 1)

def choose(n, k):
  return factoriel(n) / (factoriel(k) * factoriel(n - k))

def main():
  n = 8
  k = 4
  print(f"Value of binomial coefficient for n = 3 and k = 4 is {choose(n, k)}.")


main()