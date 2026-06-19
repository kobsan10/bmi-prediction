# BMI Prediction

Predicts a person's BMI category from height and weight using multiple linear regression computed from scratch (no scikit-learn).

## Project Structure

```
bmi-prediction/
├── bmi_prediction.py   # Main script: trains coefficients and runs a single prediction
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
- numpy
- pandas

## Installation

```bash
pip install numpy pandas
```

## Dataset

500-Person Gender-Height-Weight-Body Mass Index dataset from Kaggle.  
Source: https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex

## Usage

```bash
python bmi_prediction.py
```

The script will:
1. Compute regression coefficients from the dataset and print them.
2. Prompt for height (cm) and weight (kg), then print the predicted BMI category.

**Example session:**
```
   Gender  Height  Weight  Index
0  Male    174     96      4
...

--- Trained Coefficients (via Manual Equations) ---
b0 (Intercept)   : -2.1837
b1 (Height Coeff) : -0.0039
b2 (Weight Coeff) : 0.0597

--- Prediction Simulation ---
Enter Height (in cm): 174
Enter Weight (in kg): 96
--------------------------------------------------
Height: 170.0 cm
Weight: 72.0 kg
Predicted BMI Index: 3
Category: Overweight
--------------------------------------------------
```

## How It Works

1. Load `bmi.csv` and extract Height (X1), Weight (X2), and Index (y).
2. Compute variance and covariance terms (corrected sums of squares/cross-products).
3. Solve for slope coefficients b1 (Height) and b2 (Weight) using the closed-form two-feature normal equations.
4. Derive intercept b0 from the feature and target means.
5. Prompt the user for height and weight, run `predict_bmi()`, and print the BMI index and category.
