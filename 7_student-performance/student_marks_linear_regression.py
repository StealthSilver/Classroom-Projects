from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



df = pd.read_csv("student_marks.csv")


print("Dataset shape:", df.shape)
display(df.head())
df.info()

required_columns = ["Study_Hours", "Attendance", "Assignment_Score"]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"The dataset is missing required columns: {missing_columns}")

rows_before_cleaning = len(df)
duplicate_rows = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

missing_rows = int(df[required_columns].isna().any(axis=1).sum())
df = df.dropna(subset=required_columns).copy()

print(f"Rows before cleaning: {rows_before_cleaning}")
print(f"Duplicate rows removed: {duplicate_rows}")
print(f"Rows with missing or invalid required values removed: {missing_rows}")
print(f"Rows after cleaning: {len(df)}")
display(df[required_columns].describe())

plt.figure(figsize=(8, 5))
plt.hist(df["Assignment_Score"], bins=20, color="steelblue", edgecolor="white")
plt.title("Distribution of Assignment Scores")
plt.xlabel("Assignment score")
plt.ylabel("Number of students")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df["Study_Hours"], df["Assignment_Score"], alpha=0.6)
study_line = np.polyfit(df["Study_Hours"], df["Assignment_Score"], 1)
study_x = np.linspace(df["Study_Hours"].min(), df["Study_Hours"].max(), 100)
axes[0].plot(study_x, np.polyval(study_line, study_x), color="red")
axes[0].set_title("Study Hours and Assignment Score")
axes[0].set_xlabel("Study hours")
axes[0].set_ylabel("Assignment score")

axes[1].scatter(df["Attendance"], df["Assignment_Score"], alpha=0.6)
attendance_line = np.polyfit(df["Attendance"], df["Assignment_Score"], 1)
attendance_x = np.linspace(df["Attendance"].min(), df["Attendance"].max(), 100)
axes[1].plot(attendance_x, np.polyval(attendance_line, attendance_x), color="red")
axes[1].set_title("Attendance and Assignment Score")
axes[1].set_xlabel("Attendance (%)")
axes[1].set_ylabel("Assignment score")

plt.tight_layout()
plt.show()

correlation = df[required_columns].corr()

fig, ax = plt.subplots(figsize=(7, 5))
heatmap = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(required_columns)), labels=required_columns, rotation=25, ha="right")
ax.set_yticks(range(len(required_columns)), labels=required_columns)
for row in range(len(required_columns)):
    for column in range(len(required_columns)):
        ax.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center")
fig.colorbar(heatmap, ax=ax, label="Correlation")
ax.set_title("Correlation Between Study, Attendance, and Score")
plt.tight_layout()
plt.show()

features = ["Study_Hours", "Attendance"]
target = "Assignment_Score"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Intercept:", round(model.intercept_, 2))
for feature, coefficient in zip(features, model.coef_):
    print(f"{feature} coefficient: {coefficient:.2f}")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.3f}")

residuals = y_test - y_pred
lower = min(y_test.min(), y_pred.min())
upper = max(y_test.max(), y_pred.max())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred, alpha=0.7, color="teal")
axes[0].plot([lower, upper], [lower, upper], "r--", label="Perfect prediction")
axes[0].set_title("Actual vs. Predicted Assignment Scores")
axes[0].set_xlabel("Actual score")
axes[0].set_ylabel("Predicted score")
axes[0].legend()

axes[1].hist(residuals, bins=20, color="purple", edgecolor="white")
axes[1].set_title("Distribution of Prediction Errors")
axes[1].set_xlabel("Residual (actual - predicted)")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.7, color="darkorange")
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs. Predicted Scores")
plt.xlabel("Predicted score")
plt.ylabel("Residual (actual - predicted)")
plt.tight_layout()
plt.show()

new_student = pd.DataFrame({
    "Study_Hours": [6.0],
    "Attendance": [85],
})

predicted_score = model.predict(new_student)[0]
print("Example student:")
print("Study hours: 6.0")
print("Attendance: 85%")
print(f"Predicted assignment score: {predicted_score:.2f}")
