# 🎮 VALORANT Match Prediction System

A sophisticated machine learning system that predicts Valorant esports match outcomes using team performance data and map statistics across three major regions: EMEA, AMERICAS, and PACIFIC.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

- **Multi-Regional Support**: Predictions for EMEA, AMERICAS, and PACIFIC regions
- **Multiple ML Models**: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting
- **Interactive Prediction**: Easy-to-use command-line interface
- **Auto Model Selection**: Automatically chooses the best performing model
- **Comprehensive Analysis**: Jupyter notebook with detailed data exploration
- **Real Team Data**: Covers all major Valorant esports teams and official maps

## 📊 Supported Teams & Maps

### 🇪🇺 EMEA Region
- **Teams**: FNATIC, Team Heretics, Team Liquid, BBL Esports, Natus Vincere, FUT Esports, Team Vitality, Karmine Corp, KOI, Gentle Mates, GIANTX, Apeks

### 🇺🇸 AMERICAS Region
- **Teams**: G2 Esports, Sentinels, MIBR, Evil Geniuses, 100 Thieves, KRÜ Esports, Cloud9, NRG, LEVIATÁN, LOUD, 2Game Esports, FURIA

### 🌏 PACIFIC Region
- **Teams**: Rex Regum Qeon, Gen.G, Paper Rex, DRX, BOOM Esports, TALON, T1, Nongshim RedForce, Team Secret, DetonatioN FocusMe, ZETA DIVISION, Global Esports

### 🗺️ Available Maps
Split • Icebox • Ascent • Haven • Fracture • Lotus • Pearl

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Quick Setup
1. Clone or download this repository
2. Navigate to the `First_model` directory
3. Install dependencies using the command above
4. You're ready to go!

## 🎯 Usage

### Method 1: Interactive Prediction (Recommended)
```bash
cd Executables
python valorant_predictor.py
```

Follow the interactive prompts to:
1. Select your region (EMEA/AMERICAS/PACIFIC)
2. Choose two teams to compete
3. Select a map
4. Get your prediction with confidence levels!

### Method 2: Data Exploration
```bash
jupyter notebook "First Model.ipynb"
```

Explore the complete data analysis, model training process, and performance comparisons.

### Method 3: Custom Predictions
```bash
cd Executables
python custom_match_predictor.py
```

For advanced users who want to customize prediction parameters.

## 📈 How It Works

### 1. Data Processing
- Regional datasets are loaded and preprocessed
- Features are one-hot encoded for machine learning compatibility
- Data is split into training and testing sets

### 2. Model Training
The system trains four different machine learning models:
- **Logistic Regression**: Fast, interpretable baseline model
- **Decision Tree**: Rule-based predictions with clear decision paths
- **Random Forest**: Ensemble method for improved accuracy
- **Gradient Boosting**: Advanced ensemble technique for complex patterns

### 3. Prediction Generation
- Models evaluate team matchups based on historical performance
- Map-specific factors are considered in the analysis
- Confidence scores indicate prediction reliability

### 4. Auto Model Selection
The system automatically selects the best-performing model based on:
- Accuracy scores
- Cross-validation results
- Regional performance differences

## 🎲 Prediction Modes

### 🤖 Auto Mode (Default)
- System automatically selects the best model for your region
- Provides single prediction with highest confidence
- Recommended for quick predictions

### 📊 Manual Mode
- Shows predictions from all four models
- Displays average ensemble prediction
- Perfect for comparing model differences

## 📁 Project Structure

```
First_model/
├── 📓 First Model.ipynb          # Complete data analysis & model development
├── 📄 README.md                  # This file
├── 📄 README.txt                 # Original documentation
├── 📂 Data/                      # Regional datasets
│   ├── EMEA_test1.csv           # EMEA match data
│   ├── AMERICAS_test1.csv       # AMERICAS match data
│   └── PACIFIC_test1.csv        # PACIFIC match data
└── ⚙️ Executables/              # Prediction tools
    ├── valorant_predictor.py    # Main interactive predictor
    ├── interactive_predictor.py # Alternative interface
    └── custom_match_predictor.py # Advanced customization
```

## 🎯 Example Usage

```
VALORANT REGIONAL MATCH PREDICTOR
================================
Available Regions:
1. EMEA
2. AMERICAS  
3. PACIFIC

Select region (1-3): 1

Selected region: EMEA

MATCHUP SELECTION - EMEA
=======================
Available teams in EMEA:
 1. FNATIC
 2. Team Heretics
 3. Team Liquid
 ...

Select Team 1 (1-12): 1
Select Team 2 (1-12): 3
Enter map name: Split

🎯 PREDICTION RESULT
===================
Match: FNATIC vs Team Liquid
Map: Split
Prediction: FNATIC wins
Confidence: 65.2%
Model Used: Random Forest
```

## 📊 Model Performance

The system continuously evaluates model performance across different regions:

- **Accuracy**: Typically 60-75% depending on region and matchup
- **Cross-Validation**: 5-fold CV ensures reliable performance estimates
- **Regional Adaptation**: Models are trained separately for each region

## 🔧 Technical Details

### Machine Learning Pipeline
1. **Data Preprocessing**: One-hot encoding, feature scaling
2. **Model Training**: Multiple algorithms with cross-validation
3. **Performance Evaluation**: Accuracy, precision, recall, F1-score
4. **Prediction Generation**: Probability-based match outcome prediction

### Key Features Used
- Team performance history
- Map-specific statistics
- Head-to-head records
- Regional meta considerations

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Data Updates**: Add new match data or team information
2. **Model Improvements**: Implement new algorithms or feature engineering
3. **Bug Fixes**: Report and fix any issues you encounter
4. **Documentation**: Improve documentation and examples

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎊 Acknowledgments

- Valorant esports community for providing match data
- Riot Games for creating the amazing game we're analyzing
- Open source ML community for the fantastic tools and libraries

## 💡 Tips for Best Results

- Higher confidence percentages indicate more reliable predictions
- Different regions may favor different models based on playstyle patterns
- Consider recent team performance and roster changes
- Use the Jupyter notebook to understand model decision-making

---

**Ready to predict some matches?** 🚀

Start with `python valorant_predictor.py` and let the AI guide your esports predictions! 
