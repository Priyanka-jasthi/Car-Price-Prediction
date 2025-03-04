
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load Dataset
df = pd.read_csv(r"D:\Car Price Prediction\Cars_data.csv")

# Drop unnecessary columns
df = df.drop(columns=["car_ID", "CarName"])  # Remove ID and complex categorical column

# Convert categorical variables using One-Hot Encoding
categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 
                    'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define target variable
predict = "price"
x = df.drop(columns=[predict])
y = df[predict]

# Train-Test Split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Decision Tree
dt_params = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt = DecisionTreeRegressor(random_state=42)
gs_dt = GridSearchCV(dt, dt_params, cv=5, scoring='r2', n_jobs=-1)
gs_dt.fit(xtrain, ytrain)
best_dt = gs_dt.best_estimator_

# Hyperparameter tuning for Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestRegressor(random_state=42)
gs_rf = GridSearchCV(rf, rf_params, cv=5, scoring='r2', n_jobs=-1)
gs_rf.fit(xtrain, ytrain)
best_rf = gs_rf.best_estimator_

# Hyperparameter tuning for Gradient Boosting
gb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10]
}
gb = GradientBoostingRegressor(random_state=42)
gs_gb = RandomizedSearchCV(gb, gb_params, cv=5, scoring='r2', n_jobs=-1, n_iter=10, random_state=42)
gs_gb.fit(xtrain, ytrain)
best_gb = gs_gb.best_estimator_

# Model Comparison
models = {"Decision Tree": best_dt,
          "Random Forest": best_rf,
          "Gradient Boosting": best_gb}

results = {}

for name, model in models.items():
    predictions = model.predict(xtest)
    r2 = r2_score(ytest, predictions)
    mae = mean_absolute_error(ytest, predictions)
    results[name] = {"R² Score": r2, "MAE": mae}

# Display results
results_df = pd.DataFrame(results).T
print(results_df)

# Best Model Selection
best_model_name = max(results, key=lambda k: results[k]['R² Score'])
best_model_instance = models[best_model_name]
print(f"Best Model: {best_model_name}")

# Saving the model
joblib.dump(best_model_instance, "car_price_model.pkl") 
print("Model saved successfully!")


# Feature Importance (for tree-based models)
if hasattr(best_model_instance, "feature_importances_"):
    importances = best_model_instance.feature_importances_
    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances, y=x.columns)
    plt.title(f"Feature Importance ({best_model_name})")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature Name")
    plt.show()
else:
    print(f"Feature importance not available for {best_model_name}")
    
joblib.dump(list(xtrain.columns), "feature_columns.pkl")



