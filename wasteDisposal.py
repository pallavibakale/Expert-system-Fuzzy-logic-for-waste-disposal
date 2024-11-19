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

# High urgency when the bin is full, has a strong odor, and extreme weather conditions
rule1 = ctrl.Rule(fullness['high'] & odor['very_strong'] & moisture['saturated'] & weather['stormy'], urgency['high'])
rule2 = ctrl.Rule(fullness['high'] & (odor['strong'] | toxicity['high']) & moisture['wet'] & weather['hot'], urgency['high'])
rule3 = ctrl.Rule(odor['very_strong'] & toxicity['high'] & weather['humid'] & moisture['moderate'], urgency['high'])

# Moderate urgency when the bin is moderately full, with noticeable odor and unfavorable weather
rule4 = ctrl.Rule(fullness['medium'] & odor['moderate'] & moisture['moderate'] & (weather['rainy'] | weather['cloudy']), urgency['medium'])
rule5 = ctrl.Rule(odor['strong'] & (moisture['wet'] | toxicity['medium']) & fullness['medium'] & weather['cold'], urgency['medium'])
rule6 = ctrl.Rule((moisture['moderate'] | odor['moderate']) & toxicity['high'] & fullness['medium'] & weather['humid'], urgency['medium'])

# Low urgency for bins that are not full, with minimal odor and favorable weather conditions
rule7 = ctrl.Rule(fullness['low'] & odor['none'] & moisture['dry'] & weather['clear'], urgency['low'])
rule8 = ctrl.Rule(fullness['low'] & odor['mild'] & (moisture['slightly_moist'] | moisture['dry']) & weather['clear'], urgency['low'])

# Considerations for specific weather conditions
rule9 = ctrl.Rule(odor['moderate'] & moisture['saturated'] & weather['stormy'], urgency['high'])
rule10 = ctrl.Rule(odor['mild'] & moisture['slightly_moist'] & weather['cloudy'], urgency['low'])
rule11 = ctrl.Rule(odor['strong'] & moisture['moderate'] & weather['rainy'], urgency['medium'])
rule12 = ctrl.Rule(moisture['wet'] & fullness['high'] & weather['rainy'], urgency['high'])
rule13 = ctrl.Rule(odor['very_strong'] & moisture['wet'] & (weather['hot'] | weather['humid']), urgency['high'])

# Additional rules for special cases
rule14 = ctrl.Rule(fullness['medium'] & moisture['dry'] & odor['moderate'] & weather['cold'], urgency['low'])
rule15 = ctrl.Rule(fullness['low'] & moisture['saturated'] & weather['rainy'], urgency['medium'])
rule16 = ctrl.Rule((toxicity['medium'] | odor['moderate']) & moisture['wet'] & weather['humid'], urgency['high'])
rule17 = ctrl.Rule(odor['none'] & moisture['dry'] & weather['cold'], urgency['low'])
rule18 = ctrl.Rule(fullness['medium'] & moisture['slightly_moist'] & weather['humid'], urgency['medium'])
rule19 = ctrl.Rule(odor['moderate'] & moisture['saturated'] & fullness['medium'], urgency['high'])
rule20 = ctrl.Rule(odor['none'] & toxicity['low'] & moisture['dry'] & weather['clear'], urgency['low'])


# Add a default rule to cover all scenarios
rule_default = ctrl.Rule(~fullness['low'] | ~toxicity['low'] | ~moisture['dry'] | ~odor['none'], urgency['low'])

# Create control system and simulation
waste_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
    rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule_default
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
test_waste_management(85, 60, 95, 90, 85)  # Saturated moisture, very strong odor, and hot weather
test_waste_management(30, 10, 20, 5, 10)   # Dry conditions, low odor, and clear weather
test_waste_management(70, 80, 60, 70, 75)  # Wet moisture, strong odor, and humid weather
test_waste_management(40, 50, 30, 20, 35)  # Slightly moist with moderate odor in cold weather
test_waste_management(95, 85, 85, 80, 90)  # Very high fullness, high toxicity, and stormy weather

