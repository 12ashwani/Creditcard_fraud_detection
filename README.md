# Credit Card Fraud Detection

## Introduction
Credit cards are widely used for online purchases, but they also pose significant fraud risks. Detecting fraudulent transactions promptly is crucial for credit card companies. This project aims to build a machine-learning model to detect fraudulent credit card transactions.

## Dataset
The dataset contains transactions made by European cardholders in September 2013, including 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with frauds accounting for only 0.172% of all transactions. The dataset has 31 columns:
- **Time:** Time elapsed between this transaction and the first transaction in the dataset (in seconds).
- **V1 to V28:** Result of a PCA transformation applied to the original features (for confidentiality reasons).
- **Amount:** The transaction amount.
- **Class:** Class label, where 1 indicates a fraudulent transaction and 0 indicates a legitimate transaction.

## Data Exploration and Preprocessing
### Initial Exploration
The dataset is loaded using pandas, and initial exploration involves checking the first few rows, summary statistics, and data types of the columns. The dataset is checked for missing values and found to be complete.

### Handling Data Imbalance
To handle the imbalance, several approaches are employed:
- **Undersampling:** Reducing the number of non-fraudulent transactions to match the count of fraudulent transactions.
- **Oversampling:** Increasing the number of fraudulent transactions to match the count of non-fraudulent transactions.
- **SMOTE (Synthetic Minority Over-sampling Technique):** Creating synthetic data points to increase the number of fraudulent transactions.
- **ADASYN (Adaptive Synthetic Sampling):** Similar to SMOTE, but focuses on creating data points in regions with low density of minority class samples.

## Model Building and Evaluation
### Model Selection
Various machine learning models are considered:
- Logistic Regression
- Decision Trees
- XGBoost

### Training and Validation
The dataset is split into training and testing sets. Cross-validation techniques are used to ensure robustness and avoid overfitting.

### Performance Metrics
Key performance metrics include accuracy, precision, recall, F1-score, and ROC-AUC. A confusion matrix is used to visualize the performance of the classification models.

### Hyperparameter Tuning
Hyperparameters are tuned using GridSearchCV to find the optimal parameters.

### Choosing the Best Model
The Logistic Regression model using the SMOTE balanced dataset showed excellent performance with an ROC score of 0.99 on the train set and 0.97 on the test set. It is chosen for its simplicity, ease of interpretation, and lower computational resource requirements.

## Results and Insights
### Summary of Findings
Key insights from the data exploration phase, including significant patterns or anomalies detected.

### Feature Importance
Important features impacting the model’s decisions are identified.

### Model Performance
The final model’s performance is reported on the test set, comparing it against baseline models. The Logistic Regression model with SMOTE balancing demonstrated high recall and a strong ROC-AUC score.

## Model Deployment
### Saving the Best Model
The best-performing model is serialized using the pickle module for deployment:
```python
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
