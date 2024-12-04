from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

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
fullness['low'] = fuzz.trimf(fullness.universe, [0, 0, 50])
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

# Odor
odor['none'] = fuzz.gaussmf(odor.universe, mean=0, sigma=10)
odor['mild'] = fuzz.gaussmf(odor.universe, mean=25, sigma=10)
odor['moderate'] = fuzz.gaussmf(odor.universe, mean=50, sigma=10)
odor['strong'] = fuzz.gaussmf(odor.universe, mean=75, sigma=10)
odor['very_strong'] = fuzz.gaussmf(odor.universe, mean=100, sigma=10)

# Weather
weather['clear'] = fuzz.trapmf(weather.universe, [0, 0, 10, 20])
weather['cloudy'] = fuzz.trapmf(weather.universe, [10, 20, 30, 40])
weather['rainy'] = fuzz.trapmf(weather.universe, [30, 40, 50, 60])
weather['humid'] = fuzz.trapmf(weather.universe, [50, 60, 70, 80])
weather['stormy'] = fuzz.trapmf(weather.universe, [70, 80, 90, 100])

# Urgency
urgency['low'] = fuzz.trimf(urgency.universe, [0, 0, 50])
urgency['medium'] = fuzz.trimf(urgency.universe, [20, 50, 80])
urgency['high'] = fuzz.trimf(urgency.universe, [50, 100, 100])

# Step 3: Define fuzzy rules
rules = [
    ctrl.Rule(fullness['high'] & odor['very_strong'] & weather['stormy'] & toxicity['severe'], urgency['high']),
    ctrl.Rule(fullness['medium'] & (toxicity['very_high'] | toxicity['severe']) & (moisture['wet'] | weather['rainy']), urgency['high']),
    ctrl.Rule(fullness['low'] & odor['none'] & moisture['dry'] & weather['clear'] & toxicity['none'], urgency['low']),
    ctrl.Rule(moisture['saturated'] & fullness['medium'] & weather['cloudy'] & toxicity['moderate'], urgency['medium']),
    ctrl.Rule(odor['strong'] & fullness['high'] & (toxicity['high'] | toxicity['very_high']) & weather['clear'], urgency['high']),
    # Additional rules...
]

# Add a fallback rule
fallback_rule = ctrl.Rule(~fullness['high'] & ~toxicity['severe'] & ~moisture['saturated'] & ~odor['very_strong'], urgency['low'])
rules.append(fallback_rule)

# Step 4: Create control system
waste_ctrl = ctrl.ControlSystem(rules)
waste_sim = ctrl.ControlSystemSimulation(waste_ctrl)

# Endpoint to handle waste management prediction
@app.route('/predict', methods=['POST'])
def predict_waste_urgency():
    try:
        # Parse input JSON
        data = request.json
        fullness_level = data.get('fullness', 0)
        toxicity_level = data.get('toxicity', 0)
        moisture_level = data.get('moisture', 0)
        odor_level = data.get('odor', 0)
        weather_level = data.get('weather', 0)

        # Set simulation inputs
        waste_sim.input['fullness'] = fullness_level
        waste_sim.input['toxicity'] = toxicity_level
        waste_sim.input['moisture'] = moisture_level
        waste_sim.input['odor'] = odor_level
        waste_sim.input['weather'] = weather_level

        # Perform fuzzy computation
        waste_sim.compute()
        urgency_score = waste_sim.output['urgency']

        # Determine decision
        decision = "Empty the bin immediately!" if urgency_score >= 85 else "No immediate need to empty the bin."

        return jsonify({
            "fullness": fullness_level,
            "toxicity": toxicity_level,
            "moisture": moisture_level,
            "odor": odor_level,
            "weather": weather_level,
            "urgency_score": round(urgency_score, 2),
            "decision": decision
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
