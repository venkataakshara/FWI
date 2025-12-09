import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load your Excel/CSV file
# If it's Excel:
# df = pd.read_excel("your_file.xlsx")

# If it's CSV:
df = pd.read_csv("fire_weeather_index")    # <-- Replace with your file name

# Clean unwanted spaces in column names
df.columns = df.columns.str.strip()

print("Columns after cleaning:")
print(df.columns)

# Drop non-numeric categorical columns
df_model = df.drop(columns=["Classes", "Region"])

# Split into features (X) and target (y)
X = df_model.drop(columns=["FWI"])
y = df_model["FWI"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build the Machine Learning model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Make predictions
pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Predict using a sample from the test set
sample = X_test.iloc[0:1]
sample_prediction = model.predict(sample)

print("\nSample Input:")
print(sample)
print("Predicted FWI:", sample_prediction)
