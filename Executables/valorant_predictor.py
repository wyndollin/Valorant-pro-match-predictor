import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Scikit-Learn imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class ValorantRegionalPredictor:
    """Comprehensive Valorant match predictor for all regions"""
    
    def __init__(self):
        """Initialize the predictor with region data"""
        self.regions = {
            'EMEA': {
                'path': 'D:/python_project/Valorant/First_model/Data/EMEA_test1.csv',
                'teams': [
                    "FNATIC", "Team Heretics", "Team Liquid", "BBL Esports", "Natus Vincere",
                    "FUT Esports", "Team Vitality", "Karmine Corp", "KOI", "Gentle Mates",
                    "GIANTX", "Apeks"
                ]
            },
            'AMERICAS': {
                'path': 'D:/python_project/Valorant/First_model/Data/AMERICAS_test1.csv',
                'teams': [
                    "G2 Esports", "Sentinels", "MIBR", "Evil Geniuses", "100 Thieves",
                    "KRÜ Esports", "Cloud9", "NRG", "LEVIATÁN", "LOUD", "2Game Esports",
                    "FURIA"
                ]
            },
            'PACIFIC': {
                'path': 'D:/python_project/Valorant/First_model/Data/PACIFIC_test1.csv',
                'teams': [
                    "Rex Regum Qeon", "Gen.G", "Paper Rex", "DRX", "BOOM Esports",
                    "TALON", "T1", "Nongshim RedForce", "Team Secret", "DetonatioN FocusMe",
                    "ZETA DIVISION", "Global Esports"
                ]
            }
        }
        
        self.available_maps = ["Split", "Icebox", "Ascent", "Haven", "Fracture", "Lotus", "Pearl"]
        self.current_region = None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = None
        self.models = None
        self.trained_models = {}
        
    def select_region(self):
        """Step 1: User selects a region"""
        print("\n" + "="*60)
        print("VALORANT REGIONAL MATCH PREDICTOR")
        print("="*60)
        print("Available Regions:")
        for i, region in enumerate(self.regions.keys(), 1):
            print(f"{i}. {region}")
        
        while True:
            try:
                choice = input("\nSelect region (1-3): ").strip()
                region_choice = int(choice)
                if 1 <= region_choice <= 3:
                    self.current_region = list(self.regions.keys())[region_choice - 1]
                    break
                else:
                    print("Please enter a number between 1 and 3.")
            except ValueError:
                print("Please enter a valid number.")
        
        print(f"\nSelected region: {self.current_region}")
        return self.current_region
    
    def load_and_prepare_data(self):
        """Load and prepare data for the selected region"""
        print(f"\nLoading {self.current_region} data...")
        
        # Load data
        self.data = pd.read_csv(self.regions[self.current_region]['path'])
        print(f"Data loaded: {len(self.data)} matches")
        
        # Prepare features and target
        target = "Won"
        features = list(self.data.columns.values)
        features.remove(target)
        X = self.data[features]
        y = self.data[target]
        
        # One-hot encode features
        X_encoded = pd.get_dummies(X)
        self.feature_columns = X_encoded.columns
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        
        print(f"Data prepared: {len(self.X_train)} training samples, {len(self.X_test)} test samples")
        print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")
        
        return X_encoded, y
    
    def get_matchup_input(self):
        """Step 2: User inputs matchup details"""
        print(f"\n" + "="*60)
        print(f"MATCHUP SELECTION - {self.current_region}")
        print("="*60)
        
        available_teams = self.regions[self.current_region]['teams']
        print(f"Available teams in {self.current_region}:")
        for i, team in enumerate(available_teams, 1):
            print(f"{i:2d}. {team}")
        
        print(f"\nAvailable maps: {', '.join(self.available_maps)}")
        
        # Get team 1
        while True:
            try:
                team1_choice = input(f"\nSelect Team 1 (1-{len(available_teams)}): ").strip()
                team1_idx = int(team1_choice) - 1
                if 0 <= team1_idx < len(available_teams):
                    team1 = available_teams[team1_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_teams)}.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get team 2
        while True:
            try:
                team2_choice = input(f"Select Team 2 (1-{len(available_teams)}): ").strip()
                team2_idx = int(team2_choice) - 1
                if 0 <= team2_idx < len(available_teams):
                    team2 = available_teams[team2_idx]
                    if team2 != team1:
                        break
                    else:
                        print("Team 2 must be different from Team 1.")
                else:
                    print(f"Please enter a number between 1 and {len(available_teams)}.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get map
        while True:
            map_name = input("Enter map name: ").strip()
            if map_name in self.available_maps:
                break
            else:
                print(f"Please enter a valid map: {', '.join(self.available_maps)}")
        
        return team1, team2, map_name
    
    def train_models(self):
        """Train all models and evaluate them"""
        print("Training models...")
        
        # Define models
        self.models = {
            "Logistic Regression": LogisticRegression(random_state=1),
            "Decision Tree": DecisionTreeClassifier(random_state=1),
            "Random Forest": RandomForestClassifier(random_state=1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=1)
        }
        
        # Train and evaluate models
        model_scores = []
        
        for name, model in self.models.items():
            # Train model
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            model_scores.append({
                "model_name": name,
                "accuracy": round(accuracy, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3)
            })
            
        
        # Create scores dataframe
        scores_df = pd.DataFrame(model_scores)
        print("Model Comparison:")
        print("="*60)
        print(scores_df.to_string(index=False))
        
        return scores_df
    
    def predict_match(self, model, team1, team2, map_name):
        """Predict match outcome using a specific model"""
        # Create new match data
        new_match = pd.DataFrame({
            'map': [map_name],
            't1_team_name': [team1],
            't2_team_name': [team2]
        })
        
        # One-hot encode
        new_match_encoded = pd.get_dummies(new_match)
        
        # Ensure all columns from training data are present
        for col in self.feature_columns:
            if col not in new_match_encoded.columns:
                new_match_encoded[col] = 0
        
        # Reorder columns to match training data
        new_match_encoded = new_match_encoded[self.feature_columns]
        
        # Make prediction
        prediction = model.predict(new_match_encoded)[0]
        probability = model.predict_proba(new_match_encoded)[0]
        
        return prediction, probability
    
    def print_prediction(self, model_name, team1, team2, map_name, prediction, probability):
        """Print prediction results"""
        print(f"\n{'='*50}")
        print(f"{model_name.upper()} PREDICTION")
        print(f"{'='*50}")
        print(f"Match: {team1} vs {team2}")
        print(f"Map: {map_name}")
        print(f"Predicted Winner: {team1 if prediction == 1 else team2}")
        print(f"Confidence: {probability[1] if prediction == 1 else probability[0]:.2%}")
        print(f"\nWin Probabilities:")
        print(f"  {team1}: {probability[1]:.2%}")
        print(f"  {team2}: {probability[0]:.2%}")
        print(f"{'='*50}")
    
    def auto_select_model(self, scores_df):
        """Automatically select the best model based on accuracy"""
        best_model_name = scores_df.loc[scores_df['accuracy'].idxmax(), 'model_name']
        best_model = self.trained_models[best_model_name]
        best_accuracy = scores_df.loc[scores_df['accuracy'].idxmax(), 'accuracy']
        
        print(f"\n" + "="*60)
        print("AUTOMATIC MODEL SELECTION")
        print("="*60)
        print(f"Best model: {best_model_name} (Accuracy: {best_accuracy:.3f})")
        
        return best_model_name, best_model
    
    def run_all_predictions(self, team1, team2, map_name):
        """Run predictions with all models"""
        print(f"\n" + "="*60)
        print("ALL MODELS PREDICTIONS")
        print("="*60)
        
        predictions = []
        
        for model_name, model in self.trained_models.items():
            prediction, probability = self.predict_match(model, team1, team2, map_name)
            self.print_prediction(model_name, team1, team2, map_name, prediction, probability)
            
            predictions.append({
                'model': model_name,
                'winner': team1 if prediction == 1 else team2,
                'team1_prob': probability[1],
                'team2_prob': probability[0]
            })
        
        # Calculate average prediction
        avg_team1_prob = np.mean([p['team1_prob'] for p in predictions])
        avg_team2_prob = np.mean([p['team2_prob'] for p in predictions])
        avg_winner = team1 if avg_team1_prob > avg_team2_prob else team2
        
        print(f"\n{'='*60}")
        print("AVERAGE PREDICTION (All Models)")
        print(f"{'='*60}")
        print(f"Match: {team1} vs {team2}")
        print(f"Map: {map_name}")
        print(f"Average Predicted Winner: {avg_winner}")
        print(f"Average Win Probabilities:")
        print(f"  {team1}: {avg_team1_prob:.2%}")
        print(f"  {team2}: {avg_team2_prob:.2%}")
        print(f"{'='*60}")
        
        return predictions
    
    def main(self):
        """Main execution flow"""
        try:
            # Step 1: Select region
            region = self.select_region()
            
            # Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Get matchup input
            team1, team2, map_name = self.get_matchup_input()
            
            # Train all models
            scores_df = self.train_models()
            
            # Step 3: Model selection
            print(f"\n" + "="*60)
            print("MODEL SELECTION")
            print("="*60)
            
            while True:
                auto_choice = input("Automatically choose the best model? (yes/no): ").strip().lower()
                if auto_choice in ['yes', 'y']:
                    # Auto-select best model
                    best_model_name, best_model = self.auto_select_model(scores_df)
                    prediction, probability = self.predict_match(best_model, team1, team2, map_name)
                    self.print_prediction(best_model_name, team1, team2, map_name, prediction, probability)
                    break
                    
                elif auto_choice in ['no', 'n']:
                    # Run all models
                    self.run_all_predictions(team1, team2, map_name)
                    break
                    
                else:
                    print("Please enter 'yes' or 'no'.")
            
            print(f"\n" + "="*60)
            print("PREDICTION COMPLETE!")
            print("="*60)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please check your inputs and try again.")

if __name__ == "__main__":
    predictor = ValorantRegionalPredictor()
    predictor.main() 