# Credit Card Fraud Detection Using Machine Learning

### Project Overview
Credit card fraud is a huge challenge in finance industries. It leads to substantial financial losses and reduced customer trust. This project demonstrates an end-to-end machine learning solution to detect fraudulent credit card transactions using supervised learning techniques.

The project follows the complete machine learning lifecycle, including:
- Problem framing
- Data preprocessing and exploratory data analysis (EDA)
- Feature engineering
- Model training and evaluation
- Ethical AI and bias considerations
- Model deployment



### Business Objective
To accurately identify fraudulent credit card transactions in order to:
- Reduce financial losses
- Improve fraud response time
- Minimise disruption to legitimate customers



### Machine Learning Problem
- Problem Type: Binary Classification  
- Target Variable: Class
  - '0' = Legitimate transaction  
  - '1' = Fraudulent transaction  

Due to severe class imbalance, accuracy alone is insufficient. The project prioritises recall and ROC-AUC to ensure fraudulent transactions are effectively detected.



### Dataset
- Data Source: Kaggle - Credit Card Fraud Dataset  
- Link: https://www.kaggle.com/mlg-ulb/creditcardfraud  
- Total Transactions: 284,807  
- Fraudulent Transactions: 492 (0.17%)

### Features
- V1-V28: PCA-transformed numerical features (anonymised)
- Time: Seconds since first transaction
- Amount: Transaction amount
- Class: Fraud label



### Project Structure
fraud-detection-ml
---data/
       -creditcard.md

---models/
       -cc_fraud_detection_model.pkl

---notebook/
       -Fraud_Detection_Model.ipynb

---src/
       -Data Processing.py
       -Feature Engineering.py
       -Model Evaluation.py
       -Model Training.py

---README.md

---requirements.txt



### Data Preprocessing & EDA
Key preprocessing steps:
- Verified absence of missing values
- Removed duplicate transactions
- Scaled Time and Amount using StandardScaler
- Stratified train-test split
- Addressed class imbalance using SMOTE

EDA insights:
- Severe class imbalance confirmed
- Fraudulent transactions typically involve smaller amounts
- PCA components show partial separation between fraud and non-fraud cases



### Models Implemented
The following models were trained and evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost (Final Selected Model)



## 🏆 Final Model Selection
XGBoost was selected as the final model due to:
- Best recall (fraud detection capability)
- Strong ROC-AUC performance
- Robust handling of non-linear relationships
- Improved performance on imbalanced data



### Evaluation Metrics
Given the imbalanced nature of the dataset, the following metrics were used:
- Recall (primary focus)
- Precision
- F1-Score
- ROC-AUC

The final XGBoost model achieved the best trade-off between detecting fraudulent transactions and minimising false positives.



### Model Persistence
The trained XGBoost model was serialized and saved using Pickle to ensure reproducibility and deployment readiness.

Model file:
cc_fraud_detection_model.pkl


### Loading the Saved Model

import pickle

with open("models/cc_fraud_detection_model.pkl", "rb") as f:
    model = pickle.load(f)


SUMMARY:
# Explainability & Ethical AI
SHAP was used to interpret model predictions and feature importance

Ethical considerations addressed:
- Risk of false positives impacting customers
- Limited interpretability due to PCA-transformed features
- Absence of demographic attributes limits fairness auditing
Mitigation Strategies:
- Threshold tuning
- Consistent model retraining and imoprovement

# Deployment Readiness & Reproducibility
The saved model can be deployed scaled

To reproduce results:
- Clone the repository
- Install dependencies
- Run the notebook or scripts
- Load the saved model from the models/ directory

# Requirements
Key libraries used:
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- shap
- matplotlib

FINAL TAKE-AWAY:
This project shows how fraudulent credit card transactions can be effectively detected using machine learning. This needs proper preprossing, imbalance handling, explainability techniques, and ethical/bias considerations.
The final model is reproducible, ready for deployment, and aligned to requirements in real-world financial institution requirements
