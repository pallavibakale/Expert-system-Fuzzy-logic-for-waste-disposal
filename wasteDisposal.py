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
weather = ctrl.Antecedent(np.arange(0, 101, 1), 'weather')

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

# Odor categories
odor['none'] = fuzz.trimf(odor.universe, [0, 0, 20])
odor['mild'] = fuzz.trimf(odor.universe, [10, 25, 40])
odor['moderate'] = fuzz.trimf(odor.universe, [30, 50, 70])
odor['strong'] = fuzz.trimf(odor.universe, [60, 75, 90])
odor['very_strong'] = fuzz.trimf(odor.universe, [80, 100, 100])

# Expanded weather categories
weather['clear'] = fuzz.trimf(weather.universe, [0, 0, 20])
weather['cloudy'] = fuzz.trimf(weather.universe, [10, 30, 50])
weather['rainy'] = fuzz.trimf(weather.universe, [30, 50, 70])
weather['humid'] = fuzz.trimf(weather.universe, [50, 70, 90])
weather['hot'] = fuzz.trimf(weather.universe, [70, 80, 100])
weather['cold'] = fuzz.trimf(weather.universe, [0, 20, 40])
weather['stormy'] = fuzz.trimf(weather.universe, [60, 90, 100])

# Define output membership functions
urgency['low'] = fuzz.trimf(urgency.universe, [0, 0, 50])
urgency['medium'] = fuzz.trimf(urgency.universe, [20, 50, 80])
urgency['high'] = fuzz.trimf(urgency.universe, [50, 100, 100])

# Step 3: Define rules
rule1 = ctrl.Rule(fullness['high'] & odor['very_strong'] & weather['stormy'], urgency['high'])
rule2 = ctrl.Rule(fullness['medium'] & (toxicity['high'] | moisture['high']) & weather['rainy'], urgency['medium'])
rule3 = ctrl.Rule(fullness['low'] & odor['none'] & weather['clear'], urgency['low'])
rule4 = ctrl.Rule((moisture['medium'] | toxicity['medium']) & fullness['medium'] & weather['cloudy'], urgency['medium'])
rule5 = ctrl.Rule(odor['strong'] & (fullness['high'] | toxicity['high']) & weather['hot'], urgency['high'])
rule6 = ctrl.Rule(fullness['high'] & moisture['high'] & odor['moderate'] & weather['humid'], urgency['high'])
rule7 = ctrl.Rule(toxicity['high'] & odor['very_strong'] & weather['stormy'], urgency['high'])
rule8 = ctrl.Rule(odor['very_strong'] & moisture['high'] & weather['humid'], urgency['high'])
rule9 = ctrl.Rule(fullness['medium'] & odor['moderate'] & toxicity['medium'] & weather['cold'], urgency['medium'])
rule10 = ctrl.Rule(fullness['low'] & odor['strong'] & weather['rainy'], urgency['medium'])

# Add a default rule to cover all scenarios
rule_default = ctrl.Rule(~fullness['low'] | ~toxicity['low'] | ~moisture['low'] | ~odor['none'], urgency['low'])

# Create control system and simulation
waste_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule_default])
waste_sim = ctrl.ControlSystemSimulation(waste_ctrl)

# Step 4: Plot membership functions with proper labels
def plot_membership_functions():
    inputs = [fullness, toxicity, moisture, odor, weather, urgency]
    titles = ['Fullness', 'Toxicity', 'Moisture', 'Odor', 'Weather', 'Urgency to Empty']
    
    for i, variable in enumerate(inputs):
        plt.figure()
        variable.view()
        plt.title(f'Membership Function: {titles[i]}')
        plt.xlabel('Percentage')
        plt.ylabel('Membership Degree')
        plt.grid(True)
        plt.show()

# Step 5: Run the test cases with error handling for missing or undefined urgency outputs
def test_waste_management(fullness_level, toxicity_level, moisture_level, odor_level, weather_level):
    try:
        # Assign inputs to the simulation
        waste_sim.input['fullness'] = fullness_level
        waste_sim.input['toxicity'] = toxicity_level
        waste_sim.input['moisture'] = moisture_level
        waste_sim.input['odor'] = odor_level
        waste_sim.input['weather'] = weather_level

        # Perform fuzzy inference
        waste_sim.compute()

        # Retrieve and plot the output
        urgency_score = waste_sim.output['urgency']

        # Plot output membership function with simulation result
        plt.figure()
        urgency.view(sim=waste_sim)
        plt.title('Output: Urgency to Empty')
        plt.xlabel('Urgency (%)')
        plt.ylabel('Membership Degree')
        plt.grid(True)
        plt.show()

        # Display results
        print(f"Fullness: {fullness_level}% | Toxicity: {toxicity_level}% | Moisture: {moisture_level}% | Odor: {odor_level}% | Weather: {weather_level}%")
        print(f"Urgency to Empty: {urgency_score:.2f}%\n")

    except Exception as e:
        print("An error occurred during simulation. Please check the input values and rules.")
        print(f"Error details: {e}\n")

# Step 6: Plot membership functions for all variables
plot_membership_functions()

# Step 7: Run test cases
test_waste_management(80, 40, 60, 70, 80)  # Test with hot weather
test_waste_management(50, 20, 30, 40, 10)  # Test with clear weather
test_waste_management(90, 90, 80, 85, 95)  # Test with stormy weather
test_waste_management(30, 60, 50, 70, 40)  # Test with cold weather
