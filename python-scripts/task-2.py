import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# DEFINING INPUT AND OUTPUT
foodQuality = ctrl.Antecedent(np.arange(0, 11, 1), "Food Quality") 
serviceQuality = ctrl.Antecedent(np.arange(0, 11, 1), "Service Quality")
tipPercentage = ctrl.Consequent(np.arange(0, 50, 1), "Tip Percentage")


# DEFINING MEMBERSHIP FUNCTIONS
foodQuality["poor"] = fuzz.trimf(foodQuality.universe, [0, 2, 3])
foodQuality["average"] = fuzz.trapmf(foodQuality.universe, [2, 4, 6, 8])
foodQuality["good"] = fuzz.trimf(foodQuality.universe, [7, 8, 10])

serviceQuality["poor"] = fuzz.trimf(serviceQuality.universe, [0, 2, 3])
serviceQuality["average"] = fuzz.trapmf(serviceQuality.universe, [2, 4, 6, 8])
serviceQuality["good"] = fuzz.trimf(serviceQuality.universe, [7, 8, 10])

tipPercentage["small"] = fuzz.trimf(tipPercentage.universe, [0, 0, 13])
tipPercentage["medium"] = fuzz.trimf(tipPercentage.universe, [0, 13, 25])
tipPercentage["big"] = fuzz.trimf(tipPercentage.universe, [13, 25, 25])


# DEFINING RULES
ruleA = ctrl.Rule(serviceQuality["good"] | foodQuality["good"], tipPercentage["big"])
ruleB = ctrl.Rule(serviceQuality["average"], tipPercentage["medium"])
ruleC = ctrl.Rule(serviceQuality["poor"] & foodQuality["poor"], tipPercentage["small"])


# CREATING CONTROL SYSTEM
system = ctrl.ControlSystem([ruleA, ruleB, ruleC])


# TURNING CONTROL SYSTEM INTO SIMULATION MODE
systemSimulation = ctrl.ControlSystemSimulation(system)


# TEST
systemSimulation.input["Food Quality"] = 6.5
systemSimulation.input["Service Quality"] = 9.8
systemSimulation.compute()
systemSimulation.output["Tip Percentage"]