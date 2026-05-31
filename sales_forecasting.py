# =========================================================
# RETAIL SALES FORECASTING PROJECT
# END-TO-END MACHINE LEARNING PROJECT
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# =========================================================
# LOAD DATASET
# =========================================================

# Make sure sales_data.csv is in the same folder
df = pd.read_csv("sales_data.csv")

# =========================================================
# DISPLAY DATA
# =========================================================

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

# =========================================================
# HANDLE WALMART DATASET COLUMN NAME
# =========================================================

# Walmart dataset uses Weekly_Sales
# Rename it to Sales for easier coding

if 'Weekly_Sales' in df.columns:
    df.rename(columns={'Weekly_Sales': 'Sales'}, inplace=True)

# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\nMISSING VALUES")
print(df.isnull().sum())

# Fill missing numeric values
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# =========================================================
# CONVERT DATE COLUMN
# =========================================================

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Quarter'] = df['Date'].dt.quarter

# =========================================================
# MONTHLY SALES TREND
# =========================================================

monthly_sales = df.groupby('Month')['Sales'].sum()

plt.figure(figsize=(12,6))
monthly_sales.plot(marker='o')

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)

plt.show()

# =========================================================
# STORE WISE SALES
# =========================================================

if 'Store' in df.columns:

    store_sales = df.groupby('Store')['Sales'].sum().head(10)

    plt.figure(figsize=(12,6))

    sns.barplot(
        x=store_sales.index,
        y=store_sales.values
    )

    plt.title("Top Stores by Sales")
    plt.xlabel("Store")
    plt.ylabel("Sales")

    plt.show()

# =========================================================
# HOLIDAY SALES ANALYSIS
# =========================================================

if 'Holiday_Flag' in df.columns:

    plt.figure(figsize=(8,5))

    sns.boxplot(
        x=df['Holiday_Flag'],
        y=df['Sales']
    )

    plt.title("Holiday vs Non-Holiday Sales")

    plt.show()

# =========================================================
# CORRELATION HEATMAP
# =========================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# =========================================================
# ENCODE CATEGORICAL COLUMNS
# =========================================================

df = pd.get_dummies(df, drop_first=True)

# =========================================================
# FEATURE SELECTION
# =========================================================

X = df.drop(['Sales', 'Date'], axis=1)
y = df['Sales']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# RANDOM FOREST MODEL
# =========================================================

print("\nTRAINING RANDOM FOREST MODEL...")

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

# =========================================================
# RANDOM FOREST PREDICTIONS
# =========================================================

rf_predictions = rf_model.predict(X_test)

# =========================================================
# RANDOM FOREST EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, rf_predictions)

mse = mean_squared_error(y_test, rf_predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, rf_predictions)

print("\nRANDOM FOREST PERFORMANCE")

print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R2 SCORE :", round(r2, 2))

# =========================================================
# XGBOOST MODEL
# =========================================================

print("\nTRAINING XGBOOST MODEL...")

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# =========================================================
# XGBOOST PREDICTIONS
# =========================================================

xgb_predictions = xgb_model.predict(X_test)

# =========================================================
# XGBOOST EVALUATION
# =========================================================

xgb_r2 = r2_score(y_test, xgb_predictions)

print("\nXGBOOST R2 SCORE :", round(xgb_r2, 2))

# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

comparison = pd.DataFrame({
    'Actual Sales': y_test.values,
    'Predicted Sales': rf_predictions
})

print("\nACTUAL VS PREDICTED")
print(comparison.head())

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(10,6))

plt.scatter(y_test, rf_predictions)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
).head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title("Top 10 Important Features")

plt.show()

# =========================================================
# SAVE PREDICTIONS
# =========================================================

comparison.to_csv(
    "sales_predictions.csv",
    index=False
)

print("\nPREDICTIONS SAVED SUCCESSFULLY!")

# =========================================================
# FINAL MESSAGE
# =========================================================

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("\nRetail Sales Forecasting using Machine Learning")