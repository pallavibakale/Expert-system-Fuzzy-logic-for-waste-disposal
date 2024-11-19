# Clustering (using K-Means) and machine learning (using a decision tree) to refine the rules for the fuzzy logic system based on real-world data.

pip install scikit-fuzzy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
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

# Define fuzzy membership functions
fullness['low'] = fuzz.trimf(fullness.universe, [0, 0, 50])
fullness['medium'] = fuzz.trimf(fullness.universe, [20, 50, 80])
fullness['high'] = fuzz.trimf(fullness.universe, [50, 100, 100])

toxicity['none'] = fuzz.trimf(toxicity.universe, [0, 0, 10])
toxicity['mild'] = fuzz.trimf(toxicity.universe, [5, 15, 30])
toxicity['moderate'] = fuzz.trimf(toxicity.universe, [20, 35, 50])
toxicity['high'] = fuzz.trimf(toxicity.universe, [45, 60, 70])
toxicity['very_high'] = fuzz.trimf(toxicity.universe, [60, 75, 90])
toxicity['severe'] = fuzz.trimf(toxicity.universe, [80, 100, 100])

moisture['dry'] = fuzz.trimf(moisture.universe, [0, 0, 20])
moisture['slightly_moist'] = fuzz.trimf(moisture.universe, [10, 25, 40])
moisture['moderate'] = fuzz.trimf(moisture.universe, [30, 50, 70])
moisture['wet'] = fuzz.trimf(moisture.universe, [60, 75, 90])
moisture['saturated'] = fuzz.trimf(moisture.universe, [80, 100, 100])

odor['none'] = fuzz.trimf(odor.universe, [0, 0, 20])
odor['mild'] = fuzz.trimf(odor.universe, [10, 25, 40])
odor['moderate'] = fuzz.trimf(odor.universe, [30, 50, 70])
odor['strong'] = fuzz.trimf(odor.universe, [60, 75, 90])
odor['very_strong'] = fuzz.trimf(odor.universe, [80, 100, 100])

weather['clear'] = fuzz.trimf(weather.universe, [0, 0, 20])
weather['cloudy'] = fuzz.trimf(weather.universe, [10, 30, 50])
weather['rainy'] = fuzz.trimf(weather.universe, [30, 50, 70])
weather['humid'] = fuzz.trimf(weather.universe, [50, 70, 90])
weather['hot'] = fuzz.trimf(weather.universe, [70, 80, 100])
weather['cold'] = fuzz.trimf(weather.universe, [0, 20, 40])
weather['stormy'] = fuzz.trimf(weather.universe, [60, 90, 100])

urgency['low'] = fuzz.trimf(urgency.universe, [0, 0, 50])
urgency['medium'] = fuzz.trimf(urgency.universe, [20, 50, 80])
urgency['high'] = fuzz.trimf(urgency.universe, [50, 100, 100])

# Step 2: Example real-world data for clustering and rule generation
# Replace this with your actual dataset
data = np.array([
    [85, 90, 95, 80, 90, 2],  # High urgency
    [20, 10, 15, 5, 10, 0],   # Low urgency
    [70, 85, 60, 70, 75, 1],  # Medium urgency
    [95, 95, 85, 90, 100, 2], # High urgency
    [10, 0, 10, 0, 0, 0]      # Low urgency
])

# Inputs (features) and Output (target)
X = data[:, :-1]  # Fullness, Toxicity, Moisture, Odor, Weather
y = data[:, -1]   # Urgency: Low=0, Medium=1, High=2

# Step 3: Clustering for Rule Discovery
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
print("Cluster Centers (Rules):\n", centroids)

# Step 4: Train Decision Tree for Rule Refinement
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# Extract decision tree rules
rules = export_text(tree, feature_names=["Fullness", "Toxicity", "Moisture", "Odor", "Weather"])
print("\nDecision Tree Rules:\n", rules)

# Step 5: Define rules based on clustering and decision tree insights
rule1 = ctrl.Rule(fullness['high'] & toxicity['high'] & moisture['wet'], urgency['high'])
rule2 = ctrl.Rule(fullness['low'] & odor['none'] & weather['clear'], urgency['low'])
rule3 = ctrl.Rule(moisture['saturated'] & weather['stormy'], urgency['high'])
rule4 = ctrl.Rule(fullness['medium'] & toxicity['moderate'] & weather['rainy'], urgency['medium'])
rule5 = ctrl.Rule(fullness['low'] & odor['mild'] & weather['cloudy'], urgency['low'])

# Add fallback rule
rule_default = ctrl.Rule(~fullness['high'] & ~toxicity['severe'] & ~moisture['saturated'], urgency['medium'])

# Create control system and simulation
waste_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule_default])
waste_sim = ctrl.ControlSystemSimulation(waste_ctrl)

# Step 6: Plot membership functions
def plot_membership_functions():
    inputs = [fullness, toxicity, moisture, odor, weather, urgency]
    titles = ['Fullness', 'Toxicity', 'Moisture', 'Odor', 'Weather', 'Urgency']
    for i, variable in enumerate(inputs):
        plt.figure()
        variable.view()
        plt.title(f'Membership Function: {titles[i]}')
        plt.xlabel('Percentage')
        plt.ylabel('Membership Degree')
        plt.grid(True)
        plt.show()

# Step 7: Run test cases
def test_waste_management(fullness_level, toxicity_level, moisture_level, odor_level, weather_level):
    try:
        waste_sim.input['fullness'] = fullness_level
        waste_sim.input['toxicity'] = toxicity_level
        waste_sim.input['moisture'] = moisture_level
        waste_sim.input['odor'] = odor_level
        waste_sim.input['weather'] = weather_level

        waste_sim.compute()
        urgency_score = waste_sim.output['urgency']

        plt.figure()
        urgency.view(sim=waste_sim)
        plt.title('Output: Urgency to Empty')
        plt.xlabel('Urgency (%)')
        plt.ylabel('Membership Degree')
        plt.grid(True)
        plt.show()

        print(f"Fullness: {fullness_level}% | Toxicity: {toxicity_level}% | Moisture: {moisture_level}% | Odor: {odor_level}% | Weather: {weather_level}%")
        print(f"Urgency to Empty: {urgency_score:.2f}%\n")

    except Exception as e:
        print("An error occurred during simulation. Please check the input values and rules.")
        print(f"Error details: {e}\n")

plot_membership_functions()

# Test cases
test_waste_management(85, 70, 95, 80, 90)   
test_waste_management(20, 10, 15, 5, 10)
test_waste_management(70, 85, 60, 70, 75) 
test_waste_management(95, 95, 85, 90, 100) 
test_waste_management(10, 0, 10, 0, 0)
