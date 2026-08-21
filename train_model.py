import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset.csv")
df = df.dropna()

if "Student_ID" in df.columns:
    df = df.drop(columns=["Student_ID"])


# -----------------------------
# Encode Categorical Columns
# -----------------------------
cat_cols = [
    "Gender",
    "Academic_Level",
    "Country",
    "Most_Used_Platform",
    "Relationship_Status"
]

encoders = {}

for col in cat_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder


# -----------------------------
# Encode Yes/No Columns
# -----------------------------
df["Affects_Academic_Performance"] = df[
    "Affects_Academic_Performance"
].map({
    "Yes": 1,
    "No": 0
})

df["Conflicts_Over_Social_Media"] = df[
    "Conflicts_Over_Social_Media"
].map({
    "Yes": 1,
    "No": 0
})


# -----------------------------
# Create Addiction Classes
# -----------------------------
def addiction_class(score):
    if score <= 4:
        return 0          # Not Addicted
    elif score <= 6:
        return 1          # Mild Addiction
    else:
        return 2          # Severe Addiction


df["addiction_level"] = df["Addicted_Score"].apply(addiction_class)


# -----------------------------
# Check Class Distribution
# -----------------------------
print("\n==============================")
print("CLASS DISTRIBUTION")
print("==============================")

print(
    df["addiction_level"]
    .value_counts()
    .sort_index()
)


# -----------------------------
# Prepare Features
# -----------------------------
X = df.drop(
    columns=[
        "Addicted_Score",
        "addiction_level"
    ]
)

y = df["addiction_level"]


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# Train Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            "Not Addicted",
            "Mild Addiction",
            "Severe Addiction"
        ],
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2]
    )
)


# -----------------------------
# Feature Importance
# -----------------------------
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")

print(feature_importance)


# -----------------------------
# Save Model
# -----------------------------
joblib.dump(
    model,
    "addiction_model.pkl"
)

joblib.dump(
    list(X.columns),
    "feature_columns.pkl"
)

joblib.dump(
    encoders,
    "encoders.pkl"
)


print("\n==============================")
print("FILES SAVED")
print("==============================")

print("✓ addiction_model.pkl")
print("✓ feature_columns.pkl")
print("✓ encoders.pkl")
