# esculetin_rf_classifier.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import randint
import os

# Load dataset
file_path = os.path.join("data", "Esculetin.csv")
df = pd.read_csv(file_path)

# Encode categorical variable
label_encoder = LabelEncoder()
df['Cancer Type'] = label_encoder.fit_transform(df['Cancer Type'])

# Binarize the Viability column
threshold = df['Viability'].median()
df['Viability'] = (df['Viability'] > threshold).astype(int)

# Feature/target split
X = df[['Time', 'DoseCoumarin', 'Cancer Type']].astype(float).dropna()
y = df['Viability'].loc[X.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter search
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None] + list(np.arange(5, 25)),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(rf, param_distributions, n_iter=100, cv=3, scoring='f1', n_jobs=-1, random_state=42)
search.fit(X_train, y_train)
best_rf = search.best_estimator_
best_rf.fit(X_train, y_train)

# Evaluation
y_pred = best_rf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['Not Viable', 'Viable'], output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Save classification report heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='viridis', fmt=".2f", linewidths=1, linecolor='gray')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.tight_layout()
plt.savefig(os.path.join("output", "Classification_Report_Heatmap.tiff"), dpi=300)
plt.savefig(os.path.join("output", "Classification_Report_Heatmap.png"), dpi=300)  # Save as PNG

# Feature importance plot
importances = best_rf.feature_importances_
features = ['Time', 'Dose Coumarin (Esculetin)', 'Cancer Type']
sorted_idx = np.argsort(importances)[::-1]
sorted_features = np.array(features)[sorted_idx]
sorted_importances = importances[sorted_idx]

plt.figure(figsize=(10, 8), dpi=300)
sns.barplot(x=sorted_features, y=sorted_importances, palette='viridis')
for i, v in enumerate(sorted_importances):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=12)
plt.ylabel('Feature Importance')
plt.xlabel('Features')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join("output", "Random_Forest_Feature_Importance.tiff"), dpi=300)
plt.savefig(os.path.join("output", "Random_Forest_Feature_Importance.png"), dpi=300)  # Save as PNG
