import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/churn.csv")
# print(df.columns.tolist()) 
df.columns = df.columns.str.strip()
# print(df.columns.tolist())

# print(df.head())
# print(df.info())

df = df.dropna()

# print(df.columns)

# for col in df.columns:
#     print(col)
#     print(df[col].unique())  
#     print(len(df[col].unique()))

df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df = df.drop(['customerID'], axis=1)
x = df.drop('Churn', axis=1)
y = df['Churn']

# print(x.shape)  
# print(y.shape)  


num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
x[num_cols] = scaler.fit_transform(x[num_cols])

df.to_csv("data/processed_churn.csv", index=False)
# Processed data saved.