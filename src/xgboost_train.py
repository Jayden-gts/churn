import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost_model import get_model

df = pd.read_csv("data/processed_churn.csv")
df = df.astype(float)

X = df.drop("Churn", axis=1).values
y = df["Churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scale = len(y_train[y_train==0]) / len(y_train[y_train==1])
model = get_model(scale_pos_weight=scale)

model.fit(X_train, y_train)

model.save_model("models/xgboost_model.json")
print("Saved")
