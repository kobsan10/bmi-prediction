# BMI Prediction

Predicts a person's BMI category from height and weight using multi-class logistic regression.

## Project Structure

```
bmi-prediction/
├── bmi_prediction.py   # Main script: training, evaluation, and interactive prediction
└── bmi.csv             # Dataset: 500 samples with Gender, Height, Weight, Index
```

## BMI Categories

| Index | Category        |
|-------|-----------------|
| 0     | Extremely Weak  |
| 1     | Weak            |
| 2     | Normal          |
| 3     | Overweight      |
| 4     | Obesity         |
| 5     | Extreme Obesity |

## Requirements

- Python 3
- pandas
- scikit-learn

## Installation

```bash
pip install pandas scikit-learn
```

## Dataset

500-Person Gender-Height-Weight-Body Mass Index dataset from Kaggle.  
Source: https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex

## Usage

```bash
python bmi_prediction.py
```

The script will:
1. Train the model and print accuracy and a full classification report.
2. Start an interactive loop — enter height (cm) and weight (kg) to get a prediction. Type `quit` to exit.

**Example session:**
```
Accuracy: 0.97

--- BMI Prediction ---
Enter Height (cm) and Weight (kg) to predict BMI category.
Type 'quit' to exit.

Height (cm): 175
Weight (kg): 80

Predicted BMI Index: 3 - Overweight
```

## How It Works

1. Load `bmi.csv` and drop the Gender column (unused feature).
2. Scale Height and Weight with `StandardScaler`.
3. Split data 80/20 into train and test sets (`random_state=42`).
4. Train a `LogisticRegression` model (`max_iter=1000`).
5. Evaluate on the test set — prints accuracy and a per-class classification report.
6. Enter an interactive loop for real-time predictions on new inputs.
