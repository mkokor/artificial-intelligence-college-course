import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# DEFINING INPUT AND OUTPUT
# INPUT: service quality (0-10), food quality (0-10)
# OUTPUT: tip amount (0-25%)
foodQuality = ctrl.Antecedent(np.arange(0, 11, 1), "Food Quality")
serviceQuality = ctrl.Antecedent(np.arange(0, 11, 1), "Service Quality")
tip = ctrl.Consequent(np.arange(0, 26, 1), "Tip Percentage")


# DEFINING MEMBERSHIP FUNCTION FOR INPUT AND OUTPUT
# INPUT : 3 - poor, average, good
foodQuality.automf(3)
serviceQuality.automf(3)

tip["small"] = fuzz.trimf(tip.universe, [0, 0, 13])
tip["medium"] = fuzz.trimf(tip.universe, [0, 13, 25])
tip["big"] = fuzz.trimf(tip.universe, [13, 25, 25])


# DEFINING RULES
ruleA = ctrl.Rule(foodQuality["poor"] | serviceQuality["poor"], tip["small"])
ruleB = ctrl.Rule(serviceQuality["average"], tip["medium"])
ruleC = ctrl.Rule(serviceQuality["good"] | foodQuality["good"], tip["big"])


# CREATING CONTROL SYSTEM
controlSystem = ctrl.ControlSystem([ruleA, ruleB, ruleC])


# TURNING SYSTEM INTO SIMULATION MODE (for usage)
systemSimulation = ctrl.ControlSystemSimulation(controlSystem)


# TESTING
systemSimulation.input["Food Quality"] = 6.5
systemSimulation.input["Service Quality"] = 9.8

systemSimulation.compute()
tipPercentage = systemSimulation.output["Tip Percentage"]

print(f"FOOD QUALITY: {6.5}")
print(f"SERVICE QUALITY: {9.8}")
print(f"TIP PERCENTAGE: {tipPercentage}")