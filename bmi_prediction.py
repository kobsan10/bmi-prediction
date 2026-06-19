import numpy as np
import pandas as pd
 
df = pd.read_csv('bmi.csv')
print(df.head())
 
X1 = df["Height"].to_numpy()
X2 = df["Weight"].to_numpy()
y = df["Index"].to_numpy()
n = len(df)

# variance and covariance terms
sumofX1squares = np.sum(X1**2) - ((np.sum(X1) ** 2) / n)

sumofX2squares = np.sum(X2**2) - ((np.sum(X2) ** 2) / n)

sumofX1y = np.sum(X1 * y) - ((np.sum(X1) * np.sum(y)) / n)

sumofX2y = np.sum(X2 * y) - ((np.sum(X2) * np.sum(y)) / n) 

sumofX1X2 = np.sum(X1 * X2) - ((np.sum(X1) * np.sum(X2)) / n)

meanofy = np.mean(y)

meanofX1 = np.mean(X1)

meanofX2 = np.mean(X2)
 
# shared denominator for both slope coefficients
denominator = (sumofX1squares * sumofX2squares) - (sumofX1X2**2)

# slope coefficients
b1 = ((sumofX2squares * sumofX1y) - (sumofX1X2 * sumofX2y)) / denominator

b2 = ((sumofX1squares * sumofX2y) - (sumofX1X2 * sumofX1y)) / denominator

b0 = meanofy - (b1 * meanofX1) - (b2 * meanofX2)
 
print("--- Trained Coefficients (via Manual Equations) ---")

print(f"b0 (Intercept)   : {b0:.4f}")

print(f"b1 (Height Coeff) : {b1:.4f}")

print(f"b2 (Weight Coeff) : {b2:.4f}\n")
 

def predict_bmi(height_cm, weight_kg):

    y_pred = b0 + (b1 * height_cm) + (b2 * weight_kg)
 
    # clamp to valid index [0-5]
    predicted_index = int(np.clip(round(y_pred), 0, 5))
 
    if predicted_index == 0:

        text_result = "Extremely Weak"

    elif predicted_index == 1:

        text_result = "Weak"

    elif predicted_index == 2:

        text_result = "Normal"

    elif predicted_index == 3:

        text_result = "Overweight"

    elif predicted_index == 4:

        text_result = "Obesity"

    elif predicted_index == 5:

        text_result = "Extreme Obesity"

    else:

        text_result = "Unknown Category"
 
    print("-" * 50)

    print(f"Height: {height_cm} cm")

    print(f"Weight: {weight_kg} kg")

    print(f"Predicted BMI Index: {predicted_index}")

    print(f"Category: {text_result}")

    print("-" * 50)
 
 
print("--- Prediction Simulation ---")
 
user_height = float(input("Enter Height (in cm): "))

user_weight = float(input("Enter Weight (in kg): "))

predict_bmi(user_height,user_weight)
 