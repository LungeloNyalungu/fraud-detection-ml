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
       -creditcard.csv

---models/
       -cc_fraud_detection_model.pkl

---notebook/
       -Fraud_Detection_Model.ipynb
---src/
       -


