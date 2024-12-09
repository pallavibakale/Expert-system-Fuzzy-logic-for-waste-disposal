from flask import Flask, request, render_template
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random

app = Flask(__name__)

# Define fuzzy variables
fullness = ctrl.Antecedent(np.arange(0, 101, 1), 'fullness')
toxicity = ctrl.Antecedent(np.arange(0, 101, 1), 'toxicity')
moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'moisture')
odor = ctrl.Antecedent(np.arange(0, 101, 1), 'odor')
weather = ctrl.Antecedent(np.arange(0, 101, 1), 'weather')
urgency = ctrl.Consequent(np.arange(0, 101, 1), 'urgency')

# Membership Functions
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

odor['none'] = fuzz.gaussmf(odor.universe, mean=0, sigma=10)
odor['mild'] = fuzz.gaussmf(odor.universe, mean=25, sigma=10)
odor['moderate'] = fuzz.gaussmf(odor.universe, mean=50, sigma=10)
odor['strong'] = fuzz.gaussmf(odor.universe, mean=75, sigma=10)
odor['very_strong'] = fuzz.gaussmf(odor.universe, mean=100, sigma=10)

weather['clear'] = fuzz.trapmf(weather.universe, [0, 0, 10, 20])
weather['cloudy'] = fuzz.trapmf(weather.universe, [10, 20, 30, 40])
weather['rainy'] = fuzz.trapmf(weather.universe, [30, 40, 50, 60])
weather['humid'] = fuzz.trapmf(weather.universe, [50, 60, 70, 80])
weather['stormy'] = fuzz.trapmf(weather.universe, [70, 80, 90, 100])

urgency['low'] = fuzz.trimf(urgency.universe, [0, 0, 50])
urgency['medium'] = fuzz.trimf(urgency.universe, [20, 50, 80])
urgency['high'] = fuzz.trimf(urgency.universe, [50, 100, 100])
urgency['critical'] = fuzz.trimf(urgency.universe, [85, 100, 100])

# Rules
rule1 = ctrl.Rule(fullness['high'] & odor['very_strong'] & toxicity['severe'] & moisture['saturated'] & weather['stormy'], urgency['critical'])
rule2 = ctrl.Rule(fullness['high'] & odor['moderate'] & toxicity['moderate'] & moisture['saturated'] & weather['stormy'], urgency['critical'])
rule3 = ctrl.Rule(fullness['medium'] & odor['very_strong'] & toxicity['very_high'] & moisture['wet'] & weather['humid'], urgency['critical'])
rule4 = ctrl.Rule(fullness['high'] & odor['strong'] & toxicity['severe'] & moisture['wet'] & weather['stormy'], urgency['critical'])
rule5 = ctrl.Rule(fullness['high'] & odor['moderate'] & toxicity['severe'] & moisture['saturated'] & weather['stormy'], urgency['critical'])
rule6 = ctrl.Rule(fullness['medium'] & odor['moderate'] & (toxicity['high'] | toxicity['very_high']) & moisture['wet'] & weather['humid'], urgency['high'])
rule7 = ctrl.Rule(fullness['medium'] & odor['very_strong'] & toxicity['high'] & moisture['wet'] & weather['humid'], urgency['critical'])
rule8 = ctrl.Rule(fullness['medium'] & odor['moderate'] & toxicity['moderate'] & moisture['wet'] & weather['stormy'], urgency['high'])
rule9 = ctrl.Rule(fullness['high'] & odor['very_strong'] & toxicity['severe'] & moisture['wet'] & weather['humid'], urgency['high'])
rule10 = ctrl.Rule(fullness['high'] & odor['very_strong'] & toxicity['moderate'] & moisture['wet'] & weather['rainy'], urgency['high'])
rule11 = ctrl.Rule(fullness['high'] & odor['moderate'] & toxicity['moderate'] & moisture['wet'] & weather['cloudy'], urgency['high'])
rule12 = ctrl.Rule(fullness['medium'] & odor['very_strong'] & toxicity['moderate'] & moisture['wet'] & weather['cloudy'], urgency['high'])
rule13 = ctrl.Rule(fullness['medium'] & odor['strong'] & toxicity['moderate'] & moisture['wet'] & weather['rainy'], urgency['high'])
rule14 = ctrl.Rule(fullness['medium'] & odor['mild'] & toxicity['moderate'] & moisture['wet'] & weather['cloudy'], urgency['medium'])
rule15 = ctrl.Rule(fullness['medium'] & odor['moderate'] & toxicity['moderate'] & moisture['moderate'] & weather['rainy'], urgency['medium'])
rule16 = ctrl.Rule(fullness['medium'] & odor['strong'] & toxicity['mild'] & moisture['wet'] & weather['cloudy'], urgency['medium'])
rule17 = ctrl.Rule(fullness['medium'] & odor['moderate'] & toxicity['moderate'] & moisture['moderate'] & weather['cloudy'], urgency['medium'])
rule18 = ctrl.Rule(fullness['medium'] & odor['moderate'] & toxicity['moderate'] & moisture['wet'] & weather['cloudy'], urgency['medium'])
rule19 = ctrl.Rule(fullness['low'] & odor['moderate'] & toxicity['mild'] & moisture['moderate'] & weather['rainy'], urgency['medium'])
rule20 = ctrl.Rule(fullness['low'] & odor['mild'] & toxicity['mild'] & moisture['wet'] & weather['rainy'], urgency['medium'])
rule21 = ctrl.Rule(fullness['medium'] & odor['moderate'] & toxicity['moderate'] & moisture['saturated'] & weather['cloudy'], urgency['medium'])
rule22 = ctrl.Rule(fullness['medium'] & odor['mild'] & toxicity['mild'] & moisture['wet'] & weather['cloudy'], urgency['medium'])
rule23 = ctrl.Rule(fullness['medium'] & odor['moderate'] & toxicity['mild'] & moisture['wet'] & weather['cloudy'], urgency['medium'])
rule24 = ctrl.Rule(fullness['low'] & odor['none'] & toxicity['none'] & moisture['dry'] & weather['clear'], urgency['low'])
rule25 = ctrl.Rule(fullness['low'] & odor['mild'] & toxicity['none'] & moisture['dry'] & weather['clear'], urgency['low'])
rule26 = ctrl.Rule(fullness['low'] & odor['mild'] & toxicity['mild'] & moisture['slightly_moist'] & weather['clear'], urgency['low'])
rule27 = ctrl.Rule(fullness['low'] & odor['mild'] & toxicity['none'] & moisture['slightly_moist'] & weather['clear'], urgency['low'])
rule28 = ctrl.Rule(fullness['high'] & odor['mild'] & toxicity['moderate'] & moisture['moderate'] & weather['cloudy'], urgency['medium'])
rule29 = ctrl.Rule(fullness['high'] & odor['very_strong'] & toxicity['high'] & moisture['saturated'] & weather['stormy'], urgency['high'])
rule30 = ctrl.Rule(fullness['low'] & odor['mild'] & toxicity['moderate'] & moisture['wet'] & weather['cloudy'], urgency['medium'])
rule31 = ctrl.Rule(fullness['low'] & odor['mild'] & toxicity['mild'] & moisture['moderate'] & weather['rainy'], urgency['medium'])
rule32 = ctrl.Rule(fullness['medium'] & odor['moderate'] & toxicity['mild'] & moisture['wet'] & weather['humid'], urgency['medium'])
rule33 = ctrl.Rule(fullness['high'] & odor['very_strong'] & toxicity['moderate'] & moisture['wet'] & weather['humid'], urgency['high'])
rule34 = ctrl.Rule(fullness['low'] & odor['none'] & toxicity['mild'] & moisture['moderate'] & weather['cloudy'], urgency['low'])
rule35 = ctrl.Rule(fullness['low'] & odor['mild'] & toxicity['none'] & moisture['dry'] & weather['cloudy'], urgency['low'])
rule36 = ctrl.Rule(fullness['high'] & odor['mild'] & toxicity['none'] & moisture['dry'] & weather['cloudy'], urgency['high'])
rule37 = ctrl.Rule(fullness['high'] & odor['strong'] & toxicity['high'] & moisture['wet'] & weather['stormy'], urgency['critical'])
rule38 = ctrl.Rule(fullness['high'] & odor['strong'] & toxicity['very_high'] & moisture['wet'] & weather['humid'], urgency['critical'])

fallback_rule = ctrl.Rule(fullness['high'], urgency['high'])  # Fallback for high fullness
rules = [
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
    rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20,
    rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30,
    rule31, rule32, rule33, rule34, rule35, rule36, rule37, rule38, fallback_rule ]

# Control System
waste_ctrl = ctrl.ControlSystem(rules)
waste_sim = ctrl.ControlSystemSimulation(waste_ctrl)

# Mappings for categorical inputs
toxicity_mapping = {
    "none": 0,
    "mild": 20,
    "moderate": 35,
    "high": 55,
    "very_high": 75,
    "severe": 90
}

moisture_mapping = {
    "dry": 10,
    "slightly_moist": 30,
    "moderate": 50,
    "wet": 70,
    "saturated": 90
}

odor_mapping = {
    "none": 0,
    "mild": 25,
    "moderate": 50,
    "strong": 75,
    "very_strong": 100
}

weather_mapping = {
    "clear": 10,
    "cloudy": 30,
    "rainy": 50,
    "humid": 70,
    "stormy": 90
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Generate initial waste conditions
        existing_waste = {
            "toxicity": random.uniform(20, 70),  # Random toxicity value
            "moisture": random.uniform(20, 70)  # Random moisture value
        }

        # Retrieve user inputs
        data = request.form
        fullness_level = float(data.get('fullness', 0))
        toxicity_level = data.get('toxicity', 'none')
        moisture_level = data.get('moisture', 'dry')
        odor_level = data.get('odor', 'none')
        weather_level = data.get('weather', 'clear')

        # Combine existing and user inputs
        combined_toxicity = (existing_waste['toxicity'] + toxicity_mapping[toxicity_level]) / 2
        combined_moisture = (existing_waste['moisture'] + moisture_mapping[moisture_level]) / 2

        # Simulate fuzzy system
        waste_sim.input['fullness'] = np.clip(fullness_level, 0, 100)
        waste_sim.input['toxicity'] = np.clip(combined_toxicity, 0, 100)
        waste_sim.input['moisture'] = np.clip(combined_moisture, 0, 100)
        waste_sim.input['odor'] = np.clip(odor_mapping[odor_level], 0, 100)
        waste_sim.input['weather'] = np.clip(weather_mapping[weather_level], 0, 100)
        try:
            if fullness_level > 95:
                urgency_score = 98.5
            else:
            # Simulate fuzzy system
                waste_sim.compute()

                # Calculate urgency score
                urgency_score = waste_sim.output['urgency']
                
            decision = "Empty the bin immediately!" if urgency_score >= 85 else "No immediate need to empty the bin."

            # Render template with results
            return render_template('index.html', existing_waste=existing_waste, urgency_score=round(urgency_score, 2),
                                   decision=decision)
        except KeyError as e:
            return render_template('index.html', error=f"Error: {str(e)} - Please check your rule coverage.")
    

    # For GET request, just render the form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
