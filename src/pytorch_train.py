import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from models.pytorch_model import ChurnModel

torch.manual_seed(42)

df = pd.read_csv("data/processed_churn.csv")

X = df.drop("Churn", axis=1).values
y = df["Churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

input_size = X_train.shape[1]
model = ChurnModel(input_size)

pos_weight = torch.tensor([len(y_train[y_train==0]) / len(y_train[y_train==1])])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

epochs = 100

for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], \nLoss: {loss.item():.4f}")

with torch.no_grad():
    logits = model(X_test)
    predictions = torch.sigmoid(logits)
    predicted = (predictions > 0.5).float()
    accuracy = (predicted == y_test).float().mean()

print("Test Accuracy:", accuracy.item())

torch.save(model.state_dict(), "models/churn_model.pth")
print("Saved")