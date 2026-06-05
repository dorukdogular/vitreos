import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import pickle

df = pd.read_csv("data/features.csv")
X = df.drop(columns=["gfa"])
y = df["gfa"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average="weighted")
print("=== RandomForest ===")
print(f"Accuracy: {rf_acc:.4f}  F1: {rf_f1:.4f}")
print(classification_report(y_test, rf_pred))

xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average="weighted")
print("=== XGBoost ===")
print(f"Accuracy: {xgb_acc:.4f}  F1: {xgb_f1:.4f}")
print(classification_report(y_test, xgb_pred))

best = rf if rf_f1 >= xgb_f1 else xgb
best_name = "RandomForest" if rf_f1 >= xgb_f1 else "XGBoost"
with open("model/gfa_classifier.pkl", "wb") as f:
    pickle.dump(best, f)
print(f"Best model ({best_name}) saved to model/gfa_classifier.pkl")
