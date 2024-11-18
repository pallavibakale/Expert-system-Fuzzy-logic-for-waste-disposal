pip install scikit-fuzzy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Step 1: Define fuzzy variables
# Inputs
fullness = ctrl.Antecedent(np.arange(0, 101, 1), 'fullness')
toxicity = ctrl.Antecedent(np.arange(0, 101, 1), 'toxicity')
moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'moisture')
odor = ctrl.Antecedent(np.arange(0, 101, 1), 'odor')

# Output
urgency = ctrl.Consequent(np.arange(0, 101, 1), 'urgency')

# Step 2: Define fuzzy membership functions for inputs
fullness['low'] = fuzz.trimf(fullness.universe, [0, 0, 50])
fullness['medium'] = fuzz.trimf(fullness.universe, [20, 50, 80])
fullness['high'] = fuzz.trimf(fullness.universe, [50, 100, 100])

toxicity['low'] = fuzz.trimf(toxicity.universe, [0, 0, 50])
toxicity['medium'] = fuzz.trimf(toxicity.universe, [20, 50, 80])
toxicity['high'] = fuzz.trimf(toxicity.universe, [50, 100, 100])

moisture['low'] = fuzz.trimf(moisture.universe, [0, 0, 50])
moisture['medium'] = fuzz.trimf(moisture.universe, [20, 50, 80])
moisture['high'] = fuzz.trimf(moisture.universe, [50, 100, 100])

odor['low'] = fuzz.trimf(odor.universe, [0, 0, 50])
odor['medium'] = fuzz.trimf(odor.universe, [20, 50, 80])
odor['high'] = fuzz.trimf(odor.universe, [50, 100, 100])

urgency['low'] = fuzz.trimf(urgency.universe, [0, 0, 50])
urgency['medium'] = fuzz.trimf(urgency.universe, [20, 50, 80])
urgency['high'] = fuzz.trimf(urgency.universe, [50, 100, 100])

# Step 4: Define rules
rule1 = ctrl.Rule(fullness['high'] & odor['high'], urgency['high'])
rule2 = ctrl.Rule(fullness['medium'] & (toxicity['high'] | moisture['high']), urgency['medium'])
rule3 = ctrl.Rule(fullness['low'] & odor['low'], urgency['low'])
rule4 = ctrl.Rule((moisture['medium'] | toxicity['medium']) & fullness['medium'], urgency['medium'])
rule5 = ctrl.Rule(odor['medium'] & (fullness['high'] | toxicity['high']), urgency['high'])
rule6 = ctrl.Rule(fullness['high'] & moisture['high'], urgency['high'])
rule7 = ctrl.Rule(toxicity['high'], urgency['high'])
rule8 = ctrl.Rule(odor['high'] & moisture['high'], urgency['high'])
rule9 = ctrl.Rule(fullness['medium'] & odor['medium'] & toxicity['medium'], urgency['medium'])
rule10 = ctrl.Rule(fullness['low'] & odor['high'], urgency['medium'])

# Step 5: Create control system and simulation
waste_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])
waste_sim = ctrl.ControlSystemSimulation(waste_ctrl)

# Step 6: Plot membership functions
def plot_membership_functions():
    inputs = [fullness, toxicity, moisture, odor]
    for variable in inputs + [urgency]:
        variable.view()

# Step 7: Run the test cases and plot the results
def test_waste_management(fullness_level, toxicity_level, moisture_level, odor_level):
    waste_sim.input['fullness'] = fullness_level
    waste_sim.input['toxicity'] = toxicity_level
    waste_sim.input['moisture'] = moisture_level
    waste_sim.input['odor'] = odor_level

    # Perform fuzzy inference
    waste_sim.compute()
    urgency_score = waste_sim.output['urgency']

    # Plot the output
    urgency.view(sim=waste_sim)

    print(f"Fullness: {fullness_level}% | Toxicity: {toxicity_level}% | Moisture: {moisture_level}% | Odor: {odor_level}%")
    print(f"Urgency to Empty: {urgency_score:.2f}%\n")

# Plot membership functions for inputs and outputs
plot_membership_functions()

# Run test cases with intermediate visualizations
test_waste_management(80, 40, 60, 70)
test_waste_management(50, 20, 30, 40)
test_waste_management(90, 90, 80, 85)
test_waste_management(30, 60, 50, 70)

plt.show()
