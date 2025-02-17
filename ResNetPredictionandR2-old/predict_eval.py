import json
import torch
import re
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd

# Load test dataset (same way as during testing)
test_data_path = "./urban_data/tencities/test1.json"
with open(test_data_path, 'r') as f:
    test_data = [json.loads(line) for line in f]

# Load saved predictions
predictions_path = "predictions.json"
with open(predictions_path, 'r') as f:
    predictions = json.load(f)

# Ensure max_values are the same as training
max_values = [135.9, 5379.6, 38447.0]  # Use training max_values

# Store ground truth and predicted values
y_true = []
y_pred = []

for item, pred in zip(test_data, predictions):
    prompt = item['prompt']

    # ✅ Extract and normalize ground truth labels the same way as training
    labels = re.findall(r'\b\d+\.\d+|\b\d+\b', prompt)  # Extract exactly 3 numbers
    labels = [float(num) for i, num in enumerate(labels)]

    predicted_values = pred["predicted_values"]  # Ensure only first 3 predictions

    # Ensure both have exactly 3 values before adding to lists
    
    y_true.append(labels)  
    y_pred.append(predicted_values)  

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ✅ Compute Metrics for Each of the 3 Categories
metrics = {"Metric": ["R² Score", "MSE", "RMSE", "MAE"]}

for i in range(3):  # Loop over the 3 categories
    r2 = r2_score(y_true[:, i], y_pred[:, i])
    mse = mean_squared_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])

    metrics[f"Category {i+1}"] = [r2, mse, rmse, mae]  # Store in dict

# ✅ Convert to DataFrame and Display
metrics_df = pd.DataFrame(metrics)

print(metrics_df)

#import ace_tools as tools
#tools.display_dataframe_to_user(name="Regression Metrics (Per Category)", dataframe=metrics_df)
