from flask import Flask, request, jsonify, render_template
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Step 1: Define fuzzy variables

fullness = ctrl.Antecedent(np.arange(0, 101, 1), 'fullness')
toxicity = ctrl.Antecedent(np.arange(0, 101, 1), 'toxicity')
moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'moisture')
odor = ctrl.Antecedent(np.arange(0, 101, 1), 'odor')
weather = ctrl.Antecedent(np.arange(0, 101, 1), 'weather')

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
urgency['critical'] = fuzz.trimf(urgency.universe, [85, 100, 100])

# Step 3: Define fuzzy rules
rule1 = ctrl.Rule(fullness['high'] & odor['very_strong'] & toxicity['severe'], urgency['critical'])
rule2 = ctrl.Rule(fullness['high'] & weather['stormy'] & moisture['saturated'], urgency['critical'])
rule3 = ctrl.Rule(odor['very_strong'] & toxicity['severe'] & weather['stormy'], urgency['critical'])
rule4 = ctrl.Rule(fullness['medium'] & toxicity['very_high'] & odor['very_strong'], urgency['critical'])
rule5 = ctrl.Rule(moisture['saturated'] & fullness['high'] & toxicity['severe'], urgency['critical'])
rule6 = ctrl.Rule(fullness['medium'] & (toxicity['high'] | toxicity['very_high']) & (moisture['wet'] | moisture['saturated']), urgency['high'])
rule7 = ctrl.Rule(odor['strong'] & weather['humid'] & toxicity['high'], urgency['high'])
rule8 = ctrl.Rule(moisture['wet'] & weather['stormy'] & toxicity['moderate'], urgency['high'])
rule9 = ctrl.Rule(fullness['high'] & toxicity['severe'] & weather['stormy'], urgency['high'])
rule10 = ctrl.Rule(fullness['high'] & weather['humid'] & odor['very_strong'], urgency['high'])
rule11 = ctrl.Rule(fullness['high'] & moisture['wet'] & odor['strong'], urgency['high'])
rule12 = ctrl.Rule(fullness['medium'] & odor['very_strong'] & toxicity['high'], urgency['high'])
rule13 = ctrl.Rule(odor['strong'] & fullness['high'] & weather['rainy'], urgency['high'])
rule14 = ctrl.Rule(fullness['medium'] & moisture['wet'] & weather['cloudy'], urgency['medium'])
rule15 = ctrl.Rule(odor['moderate'] & weather['rainy'] & toxicity['moderate'], urgency['medium'])
rule16 = ctrl.Rule(fullness['medium'] & odor['strong'] & weather['humid'], urgency['medium'])
rule17 = ctrl.Rule(moisture['moderate'] & fullness['medium'] & weather['cloudy'], urgency['medium'])
rule18 = ctrl.Rule(fullness['medium'] & moisture['wet'] & odor['moderate'], urgency['medium'])
rule19 = ctrl.Rule(odor['moderate'] & toxicity['mild'] & weather['rainy'], urgency['medium'])
rule20 = ctrl.Rule(fullness['low'] & weather['rainy'] & toxicity['mild'], urgency['medium'])
rule21 = ctrl.Rule(odor['moderate'] & moisture['saturated'] & weather['cloudy'], urgency['medium'])
rule22 = ctrl.Rule(fullness['medium'] & odor['mild'] & weather['cloudy'], urgency['medium'])
rule23 = ctrl.Rule(fullness['medium'] & toxicity['mild'] & weather['humid'], urgency['medium'])
rule24 = ctrl.Rule(fullness['low'] & odor['none'] & toxicity['none'], urgency['low'])
rule25 = ctrl.Rule(moisture['dry'] & weather['clear'] & odor['none'], urgency['low'])
rule26 = ctrl.Rule(fullness['medium'] & toxicity['none'] & odor['mild'], urgency['low'])
rule27 = ctrl.Rule(moisture['dry'] & fullness['low'] & weather['clear'], urgency['low'])
rule28 = ctrl.Rule(odor['mild'] & weather['clear'] & fullness['low'], urgency['low'])
rule29 = ctrl.Rule(moisture['slightly_moist'] & fullness['low'] & toxicity['mild'], urgency['low'])
rule30 = ctrl.Rule(odor['mild'] & weather['cloudy'] & toxicity['none'], urgency['low'])
rule31 = ctrl.Rule(moisture['dry'] & fullness['low'] & odor['none'], urgency['low'])
rule32 = ctrl.Rule(fullness['low'] & odor['mild'] & toxicity['none'], urgency['low'])
rule33 = ctrl.Rule(weather['clear'] & moisture['dry'] & fullness['low'], urgency['low'])
rule34 = ctrl.Rule(fullness['high'] & toxicity['moderate'] & odor['mild'], urgency['medium'])
rule35 = ctrl.Rule(odor['very_strong'] & weather['stormy'] & toxicity['high'], urgency['high'])
rule36 = ctrl.Rule(fullness['low'] & toxicity['moderate'] & weather['cloudy'], urgency['medium'])
rule37 = ctrl.Rule(moisture['wet'] & odor['mild'] & weather['rainy'], urgency['medium'])
rule38 = ctrl.Rule(fullness['medium'] & toxicity['mild'] & odor['moderate'], urgency['medium'])
rule39 = ctrl.Rule(fullness['high'] & odor['very_strong'] & weather['humid'], urgency['high'])
rule40 = ctrl.Rule(fullness['low'] & odor['none'] & toxicity['none'], urgency['low'])
rule41 = ctrl.Rule(weather['cloudy'] & moisture['moderate'] & toxicity['mild'], urgency['low'])
rule42 = ctrl.Rule(odor['mild'] & weather['clear'] & moisture['dry'], urgency['low'])
rule43 = ctrl.Rule(weather['cloudy'] & toxicity['none'] & odor['mild'], urgency['low'])

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
        
        waste_ctrl = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
            rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20,
            rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30,
            rule31, rule32, rule33, rule34, rule35, rule36, rule37, rule38, rule39, rule40,
            rule41, rule42, rule43
        ])
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
