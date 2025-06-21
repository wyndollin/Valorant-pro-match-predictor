import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def load_and_train_model():
    """Load the EMEA data and train the logistic regression model"""
    # Load the EMEA dataset with updated path
    EMEA = pd.read_csv("D:/python_project/Valorant/First_model/Data/EMEA_test1.csv")
    
    # Prepare features and target
    target = "Won"
    features = list(EMEA.columns.values)
    features.remove(target)
    X = EMEA[features]
    y = EMEA[target]
    
    # One-hot encode the features
    X_encoded = pd.get_dummies(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Train the logistic regression model
    model = LogisticRegression(random_state=1)
    model.fit(X_train, y_train)
    
    return model, X_encoded.columns

def predict_match(model, feature_columns, team1_name, team2_name, map_name):
    """Predict the winner of a match"""
    # Create a new row with the match data
    new_match = pd.DataFrame({
        'map': [map_name],
        't1_team_name': [team1_name],
        't2_team_name': [team2_name]
    })
    
    # One-hot encode the new data
    new_match_encoded = pd.get_dummies(new_match)
    
    # Ensure all columns from training data are present
    for col in feature_columns:
        if col not in new_match_encoded.columns:
            new_match_encoded[col] = 0
    
    # Reorder columns to match training data
    new_match_encoded = new_match_encoded[feature_columns]
    
    # Make prediction
    prediction = model.predict(new_match_encoded)[0]
    probability = model.predict_proba(new_match_encoded)[0]
    
    return prediction, probability

def print_prediction(team1, team2, map_name, prediction, probability):
    """Print prediction results in a nice format"""
    print(f"\n{'='*50}")
    print(f"MATCH PREDICTION")
    print(f"{'='*50}")
    print(f"Match: {team1} vs {team2}")
    print(f"Map: {map_name}")
    print(f"Predicted Winner: {team1 if prediction == 1 else team2}")
    print(f"Confidence: {probability[1] if prediction == 1 else probability[0]:.2%}")
    print(f"\nWin Probabilities:")
    print(f"  {team1}: {probability[1]:.2%}")
    print(f"  {team2}: {probability[0]:.2%}")
    print(f"{'='*50}")

# Load and train the model
print("Loading and training the logistic regression model...")
model, feature_columns = load_and_train_model()
print("Model trained successfully!")

# ============================================================================
# ADD YOUR CUSTOM MATCHUPS HERE
# ============================================================================

# Example 1: Test a matchup that wasn't in the training data
print("\n" + "="*60)
print("CUSTOM MATCHUP PREDICTIONS")
print("="*60)

# You can modify these values to test different matchups
custom_matches = [
    ("FNATIC", "Team Heretics", "Lotus"),
    ("Team Liquid", "Natus Vincere", "Split"),
    ("BBL Esports", "Karmine Corp", "Haven"),
    ("FUT Esports", "Team Vitality", "Icebox"),
    ("GIANTX", "Apeks", "Ascent")
]

# Test each custom matchup
for team1, team2, map_name in custom_matches:
    try:
        prediction, probability = predict_match(model, feature_columns, team1, team2, map_name)
        print_prediction(team1, team2, map_name, prediction, probability)
    except Exception as e:
        print(f"\nError predicting {team1} vs {team2} on {map_name}: {e}")

# ============================================================================
# AVAILABLE TEAMS AND MAPS
# ============================================================================

print(f"\n{'='*60}")
print("AVAILABLE TEAMS AND MAPS FOR PREDICTIONS")
print(f"{'='*60}")
print("Teams: FNATIC, Team Heretics, Team Liquid, BBL Esports, Natus Vincere,")
print("       FUT Esports, Team Vitality, Karmine Corp, KOI, Gentle Mates,")
print("       GIANTX, Apeks")
print("\nMaps: Split, Icebox, Ascent, Haven, Fracture, Lotus, Pearl")
print(f"{'='*60}")

print("\nTo test your own matchups, modify the 'custom_matches' list in this script!") 