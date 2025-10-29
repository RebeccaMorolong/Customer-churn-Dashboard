import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
df = df.drop(['customerID'], axis=1)

# Encode categoricals
le = LabelEncoder()
for col in df.select_dtypes('object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)
print(f"AUC: {model.score(X_test, y_test):.2f}")

joblib.dump(model, 'models/churn_model.pkl')
joblib.dump(le, 'models/le.pkl')  