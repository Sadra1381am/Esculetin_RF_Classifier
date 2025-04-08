# Esculetin_RF_Classifier

This repository contains code and results for training a Random Forest classifier to predict binary cancer cell viability after Esculetin treatment. The model uses data from the Esculetin.csv dataset to predict whether the viability of cancer cells after Esculetin treatment is "Viable" or "Not Viable."


## Project Structure

- `esculetin_rf_classifier.py`: Main script for model training, evaluation, and visualization.
- ‍‍‍‍‍‍‍`predict.py`: Script for predicting the viability of cancer cells based on user input for Cancer Type, Time, and Dose Coumarin.
- `code/requirements.txt`: Dependencies for setting up the environment.
- `data/Esculetin.csv`: The Esculetin Dataset containing cancer type, time, dose, and viability information.
- `output/`: Directory containing saved PNG and TIFF figures (classification report heatmap, feature importance plots) and the trained model file esculetin_rf_model.pkl.

## Setup

1. Clone this repository and navigate to the project folder.

2. Install the required dependencies by running the following command:
```bash
pip install -r code/requirements.txt

3. Run the training and evaluation script:
```bash
python esculetin_rf_classifier.py

4. After training, you can use the predict.py script to make predictions:
```bash
python predict.py


The script will prompt you to input:
- Cancer Type (e.g., Liver, Breast, etc.)
- Time (duration in hours, e.g., 24.0, 48.0, etc.)
- Dose Coumarin (amount in μM, e.g., 50.0, 100.0, etc.)

Based on your input, the script will return the predicted viability for the selected conditions, formatted as:

For Liver cancer type at 24 h and 50 μM, the predicted viability is: Viable.