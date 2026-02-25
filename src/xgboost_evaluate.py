import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv("data/processed_churn.csv")
df = df.astype(float)

X = df.drop("Churn", axis=1).values
y = df["Churn"].values

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier()
model.load_model("models/xgboost_model.json")

probs = model.predict_proba(X_test)[:, 1]
predicted = (probs > 0.5).astype(int)

print("=" * 40)
print("CLASSIFICATION REPORT")
print("=" * 40)
print(classification_report(y_test, predicted, target_names=["Stay", "Churn"]))

print("CONFUSION MATRIX")
print("=" * 40)
cm = confusion_matrix(y_test, predicted)
print(f"                Predicted Stay  Predicted Churn")
print(f"Actual Stay          {cm[0][0]:<6}          {cm[0][1]:<6}")
print(f"Actual Churn         {cm[1][0]:<6}          {cm[1][1]:<6}")

print()
print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
print()
print("What this means:")
print(f"  - Of customers who actually churned, the model caught {cm[1][1]/(cm[1][0]+cm[1][1])*100:.1f}% of them (Recall)")
print(f"  - When the model predicted churn, it was right {cm[1][1]/(cm[0][1]+cm[1][1])*100:.1f}% of the time (Precision)")