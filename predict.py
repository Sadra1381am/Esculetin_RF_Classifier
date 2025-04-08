import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load dataset
data_path = os.path.join("data", "Esculetin.csv")
df = pd.read_csv(data_path)

# Encode cancer types using training mapping
label_encoder = LabelEncoder()
df['Cancer Type Encoded'] = label_encoder.fit_transform(df['Cancer Type'])

# Step 1: Select Cancer Type
unique_cancer_types = sorted(df['Cancer Type'].dropna().unique())
print("\nAvailable Cancer Types:")
for ct in unique_cancer_types:
    print(f"- {ct}")
cancer_input = input("\nEnter Cancer Type: ").strip()
assert cancer_input in unique_cancer_types, "Invalid Cancer Type."

# Filter dataset by selected Cancer Type
df_cancer = df[df['Cancer Type'] == cancer_input]

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

# Encode selected cancer type
cancer_encoded = label_encoder.transform([cancer_input])[0]

# Prepare input
X_new = pd.DataFrame([[time_input, dose_input, cancer_encoded]], columns=['Time', 'DoseCoumarin', 'Cancer Type'])

# Load trained model
model_path = os.path.join("output", "esculetin_rf_model.pkl")
model = joblib.load(model_path)

# Predict viability
prediction = model.predict(X_new)[0]
label = "Viable" if prediction == 1 else "Not Viable"

# Final output
print(f"\n🔬 For {cancer_input} cancer type at {time_input} h and {dose_input:.1f} μM, the predicted viability is: {label}.")