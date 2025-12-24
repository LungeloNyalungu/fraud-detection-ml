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
