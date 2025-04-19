import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load dataset
data_path = os.path.join("data", "Esculetin.csv")
df = pd.read_csv(data_path)

# Encode cancer types using training mapping
label_encoder = LabelEncoder()
df['Cancer Type'] = label_encoder.fit_transform(df['Cancer Type'])  # Note: Encoding the original column

# Step 1: Select Cancer Type
original_labels = label_encoder.classes_
print("\nAvailable Cancer Types:")
for ct in original_labels:
    print(f"- {ct}")
cancer_input = input("\nEnter Cancer Type: ").strip()
assert cancer_input in original_labels, "Invalid Cancer Type."

# Filter dataset by selected Cancer Type
df_cancer = df[df['Cancer Type'] == label_encoder.transform([cancer_input])[0]]

# Step 2: Select Time based on selected Cancer Type
available_times = sorted(df_cancer['Time'].dropna().unique())
print("\nAvailable Time values for selected Cancer Type:")
print(available_times)
time_input = float(input("Enter Time: "))
assert time_input in available_times, "Invalid Time value for selected Cancer Type."

# Filter further by selected Time
df_filtered = df_cancer[df_cancer['Time'] == time_input]

# Step 3: Enter Dose Coumarin (filtered bounds only)
dose_min = df_filtered['DoseCoumarin'].min()
dose_max = df_filtered['DoseCoumarin'].max()
print(f"\nEnter Dose Coumarin (Range for selected Cancer Type and Time: {dose_min} to {dose_max} μM)")
dose_input = float(input("Enter Dose Coumarin: "))
assert dose_min <= dose_input <= dose_max, "Dose Coumarin out of allowed range."

# Prepare input with EXACTLY the same feature names as during training
X_new = pd.DataFrame([[time_input, dose_input, label_encoder.transform([cancer_input])[0]]], 
                    columns=['Time', 'DoseCoumarin', 'Cancer Type'])  # Note the exact column names

# Load trained model
model_path = os.path.join("output", "esculetin_rf_model.pkl")
model = joblib.load(model_path)

# Predict
try:
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_new)[0]
        viability_percentage = proba[1] * 100  # Assuming class 1 is "Viable"
    else:
        prediction = model.predict(X_new)[0]
        viability_percentage = 100 if prediction == 1 else 0
    
    print(f"\nFor {cancer_input} cancer type at {time_input} h and {dose_input:.1f} μM,")
    print(f"Predicted viability: {viability_percentage:.1f}%")

    
except Exception as e:
    print(f"\nError during prediction: {str(e)}")
    print("Possible causes:")
    print("- The model was trained with different feature names")
    print("- The model expects different input features")
    print("\nPlease verify that the model was trained with exactly these features:")
    print(X_new.columns.tolist())