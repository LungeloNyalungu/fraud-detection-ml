### DATA PREPROCESSING & CLEANING
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import shap

df.isnull().sum()
df.duplicated().sum()
df = df.drop_duplicates()

df['Amount_log'] = np.log1p(df['Amount'])


### FEATURE ENGINEERING
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test


df["Amount_scaled"] = scaler.fit_transform(df[["Amount_log"]])
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])

df["Time_bin"] = pd.cut(
    df["Time"],
    bins=4,
    labels=["Very Early", "Early", "Mid", "Late"])
df = pd.get_dummies(df, columns=["Time_bin"], drop_first=True)

df["High_Amount"] = (df["Amount"] > 200).astype(int)

X = df.drop(columns=["Class"])
y = df["Class"]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10)

selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]


### MODEL EXPLAINING
import shap
explainer = shap.TreeExplainer(rf)
X_sampled = X.sample(1000, random_state=42)
shap_values = explainer.shap_values(X_sampled)


### MODEL TRAINING
X = df.drop(columns=["Class"])
y = df["Class"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42)

#handle imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
    tol=1e-3,
    random_state=42)
log_model.fit(X_train_res_scaled, y_train_res)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(
    max_depth=10,
    class_weight="balanced",
    random_state=42)
dt_model.fit(X_train_res, y_train_res)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced",
    random_state=42)
rf_model.fit(X_train_res, y_train_res)

#XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42)
xgb_model.fit(X_train_res, y_train_res)


### MODEL EVALUATION
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    return {
        "Model": name,
        "Accuracy": acc,
        "ROC-AUC": auc    }


