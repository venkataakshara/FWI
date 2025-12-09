import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\aksha\Downloads\infosys\fire_weather_index.csv.csv")

# Fix column names (remove spaces)
df.columns = df.columns.str.strip()
print("Cleaned Columns:", df.columns.tolist())

# Feature Selection
X = df[['Temperature', 'RH', 'Ws', 'Rain']]
y = df['Classes']

# Encode labels (fire / not fire)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
