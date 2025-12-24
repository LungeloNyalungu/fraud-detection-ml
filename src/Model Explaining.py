### MODEL EXPLAINING
import shap
explainer = shap.TreeExplainer(rf)
X_sampled = X.sample(1000, random_state=42)
shap_values = explainer.shap_values(X_sampled)
