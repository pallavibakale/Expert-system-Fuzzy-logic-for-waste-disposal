from flask import Flask, request, jsonify, render_template
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Step 1: Define fuzzy variables
# Inputs
fullness = ctrl.Antecedent(np.arange(0, 101, 1), 'fullness')
toxicity = ctrl.Antecedent(np.arange(0, 101, 1), 'toxicity')
moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'moisture')
odor = ctrl.Antecedent(np.arange(0, 101, 1), 'odor')
weather = ctrl.Antecedent(np.arange(0, 101, 1), 'weather')

# Output
urgency = ctrl.Consequent(np.arange(0, 101, 1), 'urgency')

# Step 2: Define fuzzy membership functions
# Fullness
fullness['low'] = fuzz.trimf(fullness.universe, [0, 0, 40])
fullness['medium'] = fuzz.trimf(fullness.universe, [20, 50, 80])
fullness['high'] = fuzz.trimf(fullness.universe, [50, 100, 100])

# Toxicity
toxicity['none'] = fuzz.trimf(toxicity.universe, [0, 0, 10])
toxicity['mild'] = fuzz.trimf(toxicity.universe, [5, 15, 30])
toxicity['moderate'] = fuzz.trimf(toxicity.universe, [20, 35, 50])
toxicity['high'] = fuzz.trimf(toxicity.universe, [45, 60, 70])
toxicity['very_high'] = fuzz.trimf(toxicity.universe, [60, 75, 90])
toxicity['severe'] = fuzz.trimf(toxicity.universe, [80, 100, 100])

# Moisture
moisture['dry'] = fuzz.trimf(moisture.universe, [0, 0, 20])
moisture['slightly_moist'] = fuzz.trimf(moisture.universe, [10, 25, 40])
moisture['moderate'] = fuzz.trimf(moisture.universe, [30, 50, 70])
moisture['wet'] = fuzz.trimf(moisture.universe, [60, 75, 90])
moisture['saturated'] = fuzz.trimf(moisture.universe, [80, 100, 100])

# Gaussian Membership Functions for Odor
odor['none'] = fuzz.gaussmf(odor.universe, mean=0, sigma=10)
odor['mild'] = fuzz.gaussmf(odor.universe, mean=25, sigma=10)
odor['moderate'] = fuzz.gaussmf(odor.universe, mean=50, sigma=10)
odor['strong'] = fuzz.gaussmf(odor.universe, mean=75, sigma=10)
odor['very_strong'] = fuzz.gaussmf(odor.universe, mean=100, sigma=20)

# Trapezoidal Membership Functions for Weather
weather['clear'] = fuzz.trapmf(weather.universe, [0, 0, 10, 20])
weather['cloudy'] = fuzz.trapmf(weather.universe, [10, 20, 30, 40])
weather['rainy'] = fuzz.trapmf(weather.universe, [30, 40, 50, 60])
weather['humid'] = fuzz.trapmf(weather.universe, [50, 60, 70, 80])
weather['stormy'] = fuzz.trapmf(weather.universe, [70, 80, 90, 100])

# Urgency
urgency['low'] = fuzz.trimf(urgency.universe, [0, 0, 50])
urgency['medium'] = fuzz.trimf(urgency.universe, [20, 50, 80])
urgency['high'] = fuzz.trimf(urgency.universe, [50, 100, 100])
urgency['critical'] = fuzz.trimf(urgency.universe, [85, 100, 100])

# Step 3: Define fuzzy rules
rules = [
    ctrl.Rule(fullness['high'] & odor['very_strong'] & weather['stormy'] & toxicity['severe'], urgency['high']),
    ctrl.Rule(fullness['medium'] & (toxicity['very_high'] | toxicity['severe']) & (moisture['wet'] | weather['rainy']), urgency['high']),
    ctrl.Rule(fullness['low'] & odor['none'] & moisture['dry'] & weather['clear'] & toxicity['none'], urgency['low']),
    ctrl.Rule(moisture['saturated'] & fullness['medium'] & weather['cloudy'] & toxicity['moderate'], urgency['medium']),
    ctrl.Rule(odor['strong'] & fullness['high'] & (toxicity['high'] | toxicity['very_high']) & weather['clear'], urgency['high']),
    ctrl.Rule(fullness['high'] & (moisture['saturated'] | moisture['wet']) & weather['humid'] & toxicity['high'], urgency['high']),
    ctrl.Rule(toxicity['severe'] & weather['stormy'] & odor['very_strong'], urgency['high']),
    ctrl.Rule(odor['moderate'] & (moisture['wet'] | moisture['saturated']) & weather['humid'], urgency['medium']),
    ctrl.Rule(fullness['medium'] & odor['moderate'] & toxicity['mild'] & weather['rainy'], urgency['medium']),
    ctrl.Rule(fullness['low'] & odor['strong'] & weather['rainy'] & toxicity['mild'], urgency['medium']),
    ctrl.Rule(fullness['high'] & (moisture['dry'] | moisture['slightly_moist']) & (toxicity['none'] | odor['none']) & weather['clear'], urgency['low']),
    ctrl.Rule(toxicity['severe'] & moisture['saturated'] & odor['very_strong'] & weather['stormy'], urgency['high']),
    ctrl.Rule(odor['none'] & fullness['medium'] & weather['clear'] & toxicity['none'], urgency['low']),
    ctrl.Rule(fullness['low'] & (moisture['wet'] | moisture['slightly_moist']) & (toxicity['mild'] | weather['cloudy']), urgency['low']),
    ctrl.Rule(toxicity['moderate'] & fullness['high'] & (moisture['moderate'] | weather['clear']), urgency['medium']),
    ctrl.Rule(fullness['medium'] & odor['strong'] & weather['humid'] & (toxicity['high'] | toxicity['very_high']), urgency['high']),
    ctrl.Rule(odor['very_strong'] & fullness['high'] & (toxicity['moderate'] | toxicity['high']) & weather['stormy'], urgency['high']),
    ctrl.Rule(toxicity['high'] & (moisture['dry'] | odor['none']) & weather['clear'], urgency['medium']),
    ctrl.Rule(toxicity['none'] & fullness['low'] & (odor['none'] & moisture['dry']), urgency['low']),
    ctrl.Rule(toxicity['mild'] & (odor['mild'] | odor['moderate']) & fullness['medium'] & weather['cloudy'], urgency['medium']),
    ctrl.Rule(fullness['high'] & (moisture['saturated'] | moisture['wet']) & (odor['very_strong'] | weather['humid']), urgency['high']),
    ctrl.Rule(odor['very_strong'] & toxicity['very_high'] & weather['stormy'], urgency['high']),
    ctrl.Rule(moisture['wet'] & fullness['low'] & toxicity['mild'] & weather['clear'], urgency['low']),
    ctrl.Rule(fullness['medium'] & odor['moderate'] & moisture['wet'] & (toxicity['moderate'] | weather['rainy']), urgency['medium']),
    ctrl.Rule(odor['very_strong'] & moisture['saturated'] & fullness['medium'] & (toxicity['severe'] | weather['stormy']), urgency['high']),
    ctrl.Rule(moisture['moderate'] & fullness['low'] & odor['mild'] & toxicity['mild'] & weather['cloudy'], urgency['low']),
    ctrl.Rule(moisture['saturated'] & fullness['high'] & (toxicity['severe'] | odor['very_strong']) & weather['humid'], urgency['high']),
    ctrl.Rule(odor['mild'] & fullness['medium'] & moisture['slightly_moist'] & (weather['cloudy'] | toxicity['mild']), urgency['medium']),
    ctrl.Rule(fullness['medium'] & moisture['dry'] & odor['none'] & weather['clear'], urgency['low']),
    ctrl.Rule(fullness['high'] & (moisture['wet'] | moisture['saturated']) & weather['humid'] & toxicity['high'], urgency['high']),
    ctrl.Rule((fullness['low'] | fullness['medium']) & moisture['wet'] & weather['humid'], urgency['medium']),
    ctrl.Rule(odor['mild'] & fullness['low'] & (moisture['dry'] | weather['clear']), urgency['low']),
    ctrl.Rule(odor['strong'] & fullness['medium'] & (toxicity['moderate'] | weather['humid']), urgency['medium']),
    ctrl.Rule(fullness['low'] & toxicity['none'] & moisture['slightly_moist'] & odor['mild'] & weather['clear'], urgency['low']),
    ctrl.Rule(weather['cloudy'] & fullness['medium'] & odor['mild'], urgency['medium']),
    ctrl.Rule(fullness['high'] & weather['humid'] & odor['very_strong'], urgency['high']),
    ctrl.Rule(weather['rainy'] & moisture['wet'] & fullness['medium'], urgency['medium']),
    ctrl.Rule(odor['moderate'] & moisture['saturated'] & fullness['low'] & weather['cloudy'], urgency['medium']),
    ctrl.Rule(fullness['high'] & (odor['none'] | moisture['slightly_moist']) & weather['clear'], urgency['low']),
    ctrl.Rule(moisture['moderate'] & odor['moderate'] & fullness['medium'] & weather['cloudy'], urgency['medium']),
    ctrl.Rule(fullness['high'] & odor['very_strong'] & toxicity['severe'], urgency['critical']),
    ctrl.Rule(fullness['high'] & moisture['saturated'] & weather['stormy'], urgency['critical']),
    ctrl.Rule(fullness['high'] & (odor['very_strong'] | toxicity['severe']), urgency['high'])
]

# Add a fallback rule
fallback_rule = ctrl.Rule(~fullness['high'] & ~toxicity['severe'] & ~moisture['saturated'] & ~odor['very_strong'], urgency['low'])
rules.append(fallback_rule)

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
# Step 4: Create control system
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_waste_urgency():
    try:
        # Parse input JSON
        data = request.json
        fullness_level = float(data.get('fullness', 0))
        toxicity_level = data.get('toxicity', "none")
        moisture_level = data.get('moisture', "dry")
        odor_level = data.get('odor', "none")
        weather_level = data.get('weather', "clear")

        print(f"Inputs: fullness={fullness_level}, toxicity={toxicity_level}, "
              f"moisture={moisture_level}, odor={odor_level}, weather={weather_level}")

        # Create a new simulation instance for each request
        waste_ctrl = ctrl.ControlSystem(rules)
        waste_sim = ctrl.ControlSystemSimulation(waste_ctrl)

        # Set simulation inputs
        waste_sim.input['fullness'] = fullness_level
        waste_sim.input['toxicity'] = toxicity_mapping.get(toxicity_level, 0)
        waste_sim.input['moisture'] = moisture_mapping.get(moisture_level, 0)
        waste_sim.input['odor'] = odor_mapping.get(odor_level, 0)
        waste_sim.input['weather'] = weather_mapping.get(weather_level, 0)

        # Critical condition
        if fullness_level >= 95 or toxicity_level == 'severe' or odor_level == 'very_strong':
            decision = "Critical condition detected. Empty the bin immediately!"
            return jsonify({
                "fullness": fullness_level,
                "toxicity": data.get('toxicity', "none"),
                "moisture": data.get('moisture', "dry"),
                "odor": data.get('odor', "none"),
                "weather": data.get('weather', "clear"),
                "urgency_score": 100,
                "decision": decision
            })
        else:
            # Perform fuzzy computation
            waste_sim.compute()
            urgency_score = waste_sim.output['urgency']
            print("Urgency Score:", urgency_score)

            # Determine decision
            decision = "Empty the bin immediately!" if urgency_score >= 85 else "No immediate need to empty the bin."

            return jsonify({
                "fullness": fullness_level,
                "toxicity": data.get('toxicity', "none"),
                "moisture": data.get('moisture', "dry"),
                "odor": data.get('odor', "none"),
                "weather": data.get('weather', "clear"),
                "urgency_score": round(urgency_score, 2),
                "decision": decision
            })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
