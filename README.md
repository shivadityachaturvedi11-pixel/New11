# New11
# California Housing Price Prediction (Linear Regression)
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Fetch Baseline Framework Assets
data = fetch_california_housing(as_frame=True)
df = pd.concat([data.data, data.target.rename('MedHouseVal')], axis=1)

X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train and Predict
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Print Evaluation Metrics
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")

# 4. Deliverable Bonus Milestone: Model Serialization
with open("task1_ml_linear_regression.pkl", "wb") as f:
    pickle.dump(model, f)
    
