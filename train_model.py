import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset.csv")
df = df.dropna()

df = df.drop(columns=["Student_ID"])

cat_cols = [
    "Gender",
    "Academic_Level",
    "Country",
    "Most_Used_Platform",
    "Relationship_Status"
]

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df["Affects_Academic_Performance"] = df["Affects_Academic_Performance"].map({"Yes": 1, "No": 0})
df["Conflicts_Over_Social_Media"] = df["Conflicts_Over_Social_Media"].map({"Yes": 1, "No": 0})

def addiction_class(score):
    if score <= 40:
        return 0
    elif score <= 70:
        return 1
    else:
        return 2

df["addiction_level"] = df["Addicted_Score"].apply(addiction_class)

X = df.drop(columns=["Addicted_Score", "addiction_level"])
y = df["addiction_level"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

