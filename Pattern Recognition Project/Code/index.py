import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load Data
df = pd.read_csv("Code/delhi_ncr_aqi_dataset.csv")
print("Shape:", df.shape)

# Simple EDA
print("\nMissing Values:\n", df.isnull().sum())


print("\nClass Distribution:\n", df["aqi_category"].value_counts())

sns.countplot(data = df, x = "aqi_category")
plt.title("AQI Category Distribution")
plt.show()

# Data Cleaning
df = df.drop(columns = ["datetime", "date", "aqi"])

for col in ["day_of_week", "season", "city", "station", "aqi_category"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Feature Selection (Simple)
corr = df.corr() #mutual

sns.heatmap(corr[["aqi_category"]], annot = True)
plt.title("Correlation with Target")
plt.show()

features = ["pm25", "pm10", "co", "no2", "temperature", "visibility", "wind_speed", "o3"]

X = df[features]
y = df["aqi_category"]

# Split + Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter = 1000),
    "Decision Tree": DecisionTreeClassifier(max_depth = 10),
    "Random Forest": RandomForestClassifier(n_estimators = 100)
}

# Training + Evaluation
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot = True, fmt = 'd')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    results.append({
        "Model": name,
        "Accuracy": acc
    })

# Comparison
results_df = pd.DataFrame(results)

sns.barplot(data = results_df, x = "Model", y = "Accuracy")
plt.title("Model Comparison")
plt.ylim(0, 1)
plt.show()