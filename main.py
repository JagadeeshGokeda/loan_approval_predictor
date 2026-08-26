import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
df = pd.read_csv("loan_approval_dataset.csv")
df =df[['loan_id', ' no_of_dependents', ' education', ' self_employed',
       ' income_annum', ' loan_amount', ' loan_term', ' cibil_score',
       ' residential_assets_value', ' commercial_assets_value',
       ' luxury_assets_value', ' bank_asset_value', ' loan_status']]
df.columns = df.columns.str.strip().str.lower()
y = df["loan_status"]
y = y.str.strip()
X = df.drop(["loan_id","loan_status"],axis = 1)
y = y.map({"Approved":1,"Rejected":0})
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
numeric_cols = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score',
       'residential_assets_value', 'commercial_assets_value',
       'luxury_assets_value', 'bank_asset_value',]
categ_cols = ['no_of_dependents', 'education', 'self_employed']
scaler = StandardScaler()
X_train_numeric = scaler.fit_transform(X_train[numeric_cols])
X_test_numeric = scaler.transform(X_test[numeric_cols])
encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)
X_train_cat = encoder.fit_transform(X_train[categ_cols])
X_test_cat = encoder.transform(X_test[categ_cols])
X_train_lol = np.hstack([X_train_numeric,X_train_cat])
X_test_lol = np.hstack([X_test_numeric,X_test_cat])
model = LogisticRegression()
model.fit(X_train_lol,y_train)
y_pred = model.predict(X_test_lol)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import pickle
with open("loan_model.pkl", "wb") as file:
    pickle.dump(model, file)
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
with open("encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)
print("Model, scaler and encoder saved successfully!")