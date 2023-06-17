from pyeasyga.pyeasyga import GeneticAlgorithm
import random
import numpy as np


# DEFINING CRITERIAL FUNCTION
def criterialFunction(x):
  x1 = x[0]
  x2 = x[1]
  return (-1 - np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2))) / (0.5 * (x1 ** 2 + x2 ** 2) + 2)

RANGE = np.array([[-5.12, 5.12], [-5.12, 5.12]])


# CREATING GENETIC ALGORITHM INSTANCE
geneticAlgorithm = GeneticAlgorithm(seed_data = criterialFunction,
                                    population_size = 30,
                                    generations = 20,
                                    crossover_probability = 0.8,
                                    mutation_probability = 0.02,
                                    elitism = True,
                                    maximise_fitness = False)

BINARY_CHROMOSOME_LENGTH = 26


# RANDOMLY GENERATING INDIVIDUAL
def createIndividual(data):
  return [random.randint(0, 1) for _ in range(BINARY_CHROMOSOME_LENGTH)]

geneticAlgorithm.create_individual = createIndividual


# CROSSOVER IN TWO POINTS
def crossover(parent1, parent2):
  crossoverIndex1 = random.randrange(1, len(parent1) - 1)
  crossoverIndex2 = random.randrange(crossoverIndex1 + 1, len(parent2))
  child1 = parent1[:crossoverIndex1] + parent2[crossoverIndex1:crossoverIndex2] + parent1[crossoverIndex2:]
  child2 = parent2[:crossoverIndex1] + parent1[crossoverIndex1:crossoverIndex2] + parent2[crossoverIndex2:]
  return child1, child2

geneticAlgorithm.crossover_function = crossover


# MUTATION AT RANDOM INDEX
def mutate(individual):
  mutationIndex = random.randrange(len(individual))
  if individual[mutationIndex] == 0:
    individual[mutationIndex] = 1
  else:
    individual[mutationIndex] = 0

geneticAlgorithm.mutate_function = mutate


# USING TOURNAMENT SELECTION BY DEFAULT
geneticAlgorithm.tournament_size = 2


# CONVERTING

# Binary to decimal...
def decimal(binary):
  sum = 0
  for i in range(0, len(binary)):
    sum += binary[len(binary) - 1 - i] * (2 ** i)
  return sum

# Chromosome to coordinates...
def decode(individual):
  x1Binary = individual[:BINARY_CHROMOSOME_LENGTH // 2]
  x2Binary = individual[BINARY_CHROMOSOME_LENGTH // 2:]
  x1Range = RANGE[0]
  x2Range = RANGE[1]
  x1Min = x1Range[0]
  x1Max = x1Range[1]
  x2Min = x2Range[0]
  x2Max = x2Range[1]
  x1Decode = x1Min + ((x1Max - x1Min) * decimal(x1Binary)) / (2 ** len(x1Binary) - 1)
  x2Decode = x2Min + ((x2Max - x2Min) * decimal(x2Binary)) / (2 ** len(x2Binary) - 1)
  return x1Decode, x2Decode


# CHECKING IF A VALUE IS IN RANGE
def isInRange(x, xRange):
  return x >= xRange[0] and x <= xRange[1]


# CREATING FITNESS FUNCTION
def fitnessFunction(individual, data):
  xDecode, yDecode = decode(individual)
  while not(isInRange(xDecode, RANGE[0]) and isInRange(yDecode, RANGE[1])):
    individual = createIndividual(data)
    xDecode, yDecode = decode(individual)
  fitness = data([xDecode, yDecode])
  return fitness

geneticAlgorithm.fitness_function = fitnessFunction


# RUN AND ANALYZING
geneticAlgorithm.run()

bestIndividualFitnessValue = geneticAlgorithm.best_individual()[0]
bestIndividualBinaryCode = geneticAlgorithm.best_individual()[1]
bestIndividualDecoded = decode(bestIndividualBinaryCode)
bestIndividualCriteriaFunction = criterialFunction(bestIndividualDecoded)

print(f"Best individual fitness value: {bestIndividualFitnessValue}")
print(f"Best individual binary code: {bestIndividualBinaryCode}")
print(f"Best individual decoded: {bestIndividualDecoded}")
print(f"Best individual criteria function value: {bestIndividualCriteriaFunction}")

print(f"Last generation: ")
for individual in geneticAlgorithm.last_generation():
  print(individual)