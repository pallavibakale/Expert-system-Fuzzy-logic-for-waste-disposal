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

# Expanded toxicity categories
toxicity['none'] = fuzz.trimf(toxicity.universe, [0, 0, 10])
toxicity['mild'] = fuzz.trimf(toxicity.universe, [5, 15, 30])
toxicity['moderate'] = fuzz.trimf(toxicity.universe, [20, 35, 50])
toxicity['high'] = fuzz.trimf(toxicity.universe, [45, 60, 70])
toxicity['very_high'] = fuzz.trimf(toxicity.universe, [60, 75, 90])
toxicity['severe'] = fuzz.trimf(toxicity.universe, [80, 100, 100])


# Expanded moisture categories
moisture['dry'] = fuzz.trimf(moisture.universe, [0, 0, 20])
moisture['slightly_moist'] = fuzz.trimf(moisture.universe, [10, 25, 40])
moisture['moderate'] = fuzz.trimf(moisture.universe, [30, 50, 70])
moisture['wet'] = fuzz.trimf(moisture.universe, [60, 75, 90])
moisture['saturated'] = fuzz.trimf(moisture.universe, [80, 100, 100])

# Expanded odor categories
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

# Step 3: Define rules considering the expanded weather and odor variable

# Adjusted and additional rules to accommodate expanded inputs
rule1 = ctrl.Rule(fullness['high'] & odor['very_strong'] & weather['stormy'] & toxicity['severe'], urgency['high'])
rule2 = ctrl.Rule(fullness['medium'] & (toxicity['very_high'] | toxicity['severe']) & (moisture['wet'] | weather['rainy']), urgency['high'])
rule3 = ctrl.Rule(fullness['low'] & odor['none'] & moisture['dry'] & weather['clear'] & toxicity['none'], urgency['low'])
rule4 = ctrl.Rule(moisture['saturated'] & fullness['medium'] & weather['cloudy'] & toxicity['moderate'], urgency['medium'])
rule5 = ctrl.Rule(odor['strong'] & fullness['high'] & (toxicity['high'] | toxicity['very_high']) & weather['hot'], urgency['high'])
rule6 = ctrl.Rule(fullness['high'] & (moisture['saturated'] | moisture['wet']) & weather['humid'] & toxicity['high'], urgency['high'])
rule7 = ctrl.Rule(toxicity['severe'] & weather['stormy'] & odor['very_strong'], urgency['high'])
rule8 = ctrl.Rule(odor['moderate'] & (moisture['wet'] | moisture['saturated']) & weather['humid'], urgency['medium'])
rule9 = ctrl.Rule(fullness['medium'] & odor['moderate'] & toxicity['mild'] & weather['cold'], urgency['medium'])
rule10 = ctrl.Rule(fullness['low'] & odor['strong'] & weather['rainy'] & toxicity['mild'], urgency['medium'])
rule11 = ctrl.Rule(fullness['high'] & (moisture['dry'] | moisture['slightly_moist']) & (toxicity['none'] | odor['none']), urgency['low'])
rule12 = ctrl.Rule(toxicity['severe'] & moisture['saturated'] & odor['very_strong'] & weather['stormy'], urgency['high'])
rule13 = ctrl.Rule(odor['none'] & fullness['medium'] & weather['clear'] & toxicity['none'], urgency['low'])
rule14 = ctrl.Rule(fullness['low'] & (moisture['wet'] | moisture['slightly_moist']) & (toxicity['mild'] | weather['cloudy']), urgency['low'])
rule15 = ctrl.Rule(toxicity['moderate'] & fullness['high'] & (moisture['moderate'] | weather['hot']), urgency['medium'])
rule16 = ctrl.Rule(fullness['medium'] & odor['strong'] & weather['humid'] & (toxicity['high'] | toxicity['very_high']), urgency['high'])
rule17 = ctrl.Rule(odor['very_strong'] & fullness['high'] & (toxicity['moderate'] | toxicity['high']) & weather['stormy'], urgency['high'])
rule18 = ctrl.Rule(toxicity['high'] & (moisture['dry'] | odor['none']) & weather['clear'], urgency['medium'])
rule19 = ctrl.Rule(toxicity['none'] & fullness['low'] & (odor['none'] & moisture['dry']), urgency['low'])
rule20 = ctrl.Rule(toxicity['mild'] & (odor['mild'] | odor['moderate']) & fullness['medium'] & weather['cold'], urgency['medium'])
rule21 = ctrl.Rule(fullness['high'] & (moisture['saturated'] | moisture['wet']) & (odor['very_strong'] | weather['humid']), urgency['high'])
rule22 = ctrl.Rule(odor['very_strong'] & toxicity['very_high'] & weather['stormy'], urgency['high'])
rule23 = ctrl.Rule(moisture['wet'] & fullness['low'] & toxicity['mild'] & weather['clear'], urgency['low'])
rule24 = ctrl.Rule(fullness['medium'] & odor['moderate'] & moisture['wet'] & (toxicity['moderate'] | weather['rainy']), urgency['medium'])
rule25 = ctrl.Rule(odor['very_strong'] & moisture['saturated'] & fullness['medium'] & (toxicity['severe'] | weather['stormy']), urgency['high'])
rule26 = ctrl.Rule(moisture['moderate'] & fullness['low'] & odor['mild'] & toxicity['mild'], urgency['low'])
rule27 = ctrl.Rule(moisture['saturated'] & fullness['high'] & (toxicity['severe'] | odor['very_strong']), urgency['high'])
rule28 = ctrl.Rule(odor['mild'] & fullness['medium'] & moisture['slightly_moist'] & (weather['cloudy'] | toxicity['mild']), urgency['medium'])
rule29 = ctrl.Rule(fullness['medium'] & moisture['dry'] & odor['none'] & weather['clear'], urgency['low'])
rule30 = ctrl.Rule(fullness['high'] & (moisture['wet'] | moisture['saturated']) & weather['hot'] & toxicity['high'], urgency['high'])
rule31 = ctrl.Rule((fullness['low'] | fullness['medium']) & moisture['wet'] & weather['humid'], urgency['medium'])
rule32 = ctrl.Rule(odor['mild'] & fullness['low'] & (moisture['dry'] | weather['clear']), urgency['low'])
rule33 = ctrl.Rule(odor['strong'] & fullness['medium'] & (toxicity['moderate'] | weather['humid']), urgency['medium'])
rule34 = ctrl.Rule(fullness['low'] & toxicity['none'] & moisture['slightly_moist'] & odor['mild'], urgency['low'])
rule35 = ctrl.Rule(weather['cold'] & fullness['medium'] & odor['mild'], urgency['medium'])
rule36 = ctrl.Rule(fullness['high'] & weather['hot'] & odor['very_strong'], urgency['high'])
rule37 = ctrl.Rule(weather['rainy'] & moisture['wet'] & fullness['medium'], urgency['medium'])
rule38 = ctrl.Rule(odor['moderate'] & moisture['saturated'] & fullness['low'], urgency['medium'])
rule39 = ctrl.Rule(fullness['high'] & (odor['none'] | moisture['slightly_moist']) & weather['clear'], urgency['low'])
rule40 = ctrl.Rule(moisture['moderate'] & odor['moderate'] & fullness['medium'] & weather['cloudy'], urgency['medium'])


# Add a comprehensive fallback rule to cover all remaining cases
rule_default = ctrl.Rule(~fullness['high'] & ~toxicity['severe'] & ~moisture['saturated'] & ~odor['very_strong'], urgency['high'])

# Create control system and simulation with the expanded rule set
waste_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
    rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20,
    rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30,
    rule31, rule32, rule33, rule34, rule35, rule36, rule37, rule38, rule39, rule40, rule_default
])
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
test_waste_management(85, 70, 95, 80, 90)
test_waste_management(20, 10, 15, 5, 10)
test_waste_management(70, 85, 60, 70, 75)
test_waste_management(95, 95, 85, 90, 100)
test_waste_management(10, 0, 10, 0, 0)