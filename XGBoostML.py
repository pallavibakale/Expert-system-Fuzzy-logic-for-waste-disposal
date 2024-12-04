import numpy as np
import pandas as pd
from itertools import product
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# Step 1: Define fuzzy input categories
moisture_levels = ['dry', 'slightly_moist', 'moderate', 'wet', 'saturated']
toxicity_levels = ['none', 'mild', 'moderate', 'high', 'very_high', 'severe']
fullness_levels = ['low', 'medium', 'high']
weather_conditions = ['clear', 'cloudy', 'rainy', 'humid', 'stormy']
odor_levels = ['none', 'mild', 'moderate', 'strong', 'very_strong']

# Map categories to numerical values
moisture_map = {level: i for i, level in enumerate(moisture_levels)}
toxicity_map = {level: i for i, level in enumerate(toxicity_levels)}
fullness_map = {level: i for i, level in enumerate(fullness_levels)}
weather_map = {level: i for i, level in enumerate(weather_conditions)}
odor_map = {level: i for i, level in enumerate(odor_levels)}

# Step 2: Generate all combinations of inputs
combinations = list(product(moisture_levels, toxicity_levels, fullness_levels, weather_conditions, odor_levels))

# Convert combinations to numerical values
data = []
for combination in combinations:
    data.append([
        moisture_map[combination[0]],
        toxicity_map[combination[1]],
        fullness_map[combination[2]],
        weather_map[combination[3]],
        odor_map[combination[4]]
    ])

# Step 3: Define target urgency levels (heuristically or manually assigned)
data = np.array(data)
urgency = (
    data[:, 0] * 0.2 +  # moisture weight
    data[:, 1] * 0.3 +  # toxicity weight
    data[:, 2] * 0.4 +  # fullness weight
    data[:, 3] * 0.2 +  # weather weight
    data[:, 4] * 0.3    # odor weight
)

# Normalize urgency to [0, 100]
urgency = (urgency / max(urgency)) * 100

# Create a DataFrame
df = pd.DataFrame(data, columns=['moisture', 'toxicity', 'fullness', 'weather', 'odor'])
df['urgency'] = urgency

# Step 4: Split data into training and testing sets
X = df.drop(columns=['urgency'])
y = df['urgency']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train an XGBoost model
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.2],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Step 8: Predict urgency based on user input
def get_user_input_and_predict():
    print("\nEnter the input levels for the following variables (choose one of the options shown):")
    
    user_moisture = input(f"Moisture ({', '.join(moisture_levels)}): ").strip().lower()
    user_toxicity = input(f"Toxicity ({', '.join(toxicity_levels)}): ").strip().lower()
    user_fullness = input(f"Fullness ({', '.join(fullness_levels)}): ").strip().lower()
    user_weather = input(f"Weather ({', '.join(weather_conditions)}): ").strip().lower()
    user_odor = input(f"Odor ({', '.join(odor_levels)}): ").strip().lower()

    # Map inputs to numerical values
    if user_moisture not in moisture_map or user_toxicity not in toxicity_map or \
       user_fullness not in fullness_map or user_weather not in weather_map or \
       user_odor not in odor_map:
        print("Invalid input. Please enter valid options.")
        return

    user_input = pd.DataFrame({
        'moisture': [moisture_map[user_moisture]],
        'toxicity': [toxicity_map[user_toxicity]],
        'fullness': [fullness_map[user_fullness]],
        'weather': [weather_map[user_weather]],
        'odor': [odor_map[user_odor]]
    })

    # Predict urgency
    predicted_urgency = best_model.predict(user_input)[0]

    # Decide if the bin should be emptied (threshold: 70%)
    threshold = 70
    decision = "Empty the bin" if predicted_urgency >= threshold else "Do not empty the bin"

    print(f"\nPredicted Urgency: {predicted_urgency:.2f}%")
    print(f"Decision: {decision}")

# Get user input and predict
get_user_input_and_predict()
