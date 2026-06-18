import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

BMI_LABELS = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity",
}

# --- Load & prepare data ---
df = pd.read_csv("bmi.csv")
df = df.drop("Gender", axis=1)

X = df[["Height", "Weight"]]
y = df["Index"]

# --- Feature scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train / test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print(classification_report(y_test, y_pred, target_names=list(BMI_LABELS.values()), zero_division=0))

# --- Interactive prediction loop ---
print("\n--- BMI Prediction ---")
print("Enter Height (cm) and Weight (kg) to predict BMI category.")
print("Type 'quit' to exit.\n")

while True:
    height_input = input("Height (cm): ").strip()
    if height_input.lower() == "quit":
        break
    weight_input = input("Weight (kg): ").strip()
    if weight_input.lower() == "quit":
        break

    try:
        height = float(height_input)
        weight = float(weight_input)
    except ValueError:
        print("Please enter valid numbers.\n")
        continue

    sample = scaler.transform(pd.DataFrame([[height, weight]], columns=["Height", "Weight"]))
    index = model.predict(sample)[0]
    print(f"\nPredicted BMI Index: {index} - {BMI_LABELS[index]}\n")
