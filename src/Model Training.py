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
