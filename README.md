# 📈 LendRisk-Predictor
Advanced Credit Default Forecasting & Risk Quantization Pipeline

## 📌 Project Vision
LendRisk-Predictor is a high-fidelity machine learning system designed to quantify the probability of financial delinquency. By analyzing historical credit utilization, debt ratios, and repayment behaviors, the system provides a calibrated risk score that assists financial institutions in making data-driven lending decisions.

This project specifically tackles the Class Imbalance Problem—a common hurdle in FinTech where defaults are rare but high-impact events—using synthetic oversampling and recursive feature pruning.

## 🛠️ Technical Implementation
Targeted Oversampling: Implemented SMOTE to balance the training distribution (approx. 6.7% default rate), ensuring the model identifies risky patterns without bias toward the majority class.

Algorithmic Feature Pruning: Used Recursive Feature Elimination (RFE) with a Random Forest backbone to isolate the 7 most predictive financial indicators, reducing model complexity and preventing overfitting.

XGBoost Architecture: Leveraged Gradient Boosting optimized with GridSearchCV. The pipeline is configured for NVIDIA CUDA acceleration, utilizing local GPU cores for high-speed hyperparameter tuning.

Probability Calibration: Beyond binary classification, the model outputs a probability spectrum, allowing for nuanced risk-tiering (e.g., Low, Medium, High Risk).

## 📊 Evaluation & Insights
The model is evaluated through a comprehensive 2x2 Dashboard:

Normalized Confusion Matrix: Visualizing the hit rate for defaults.

Precision-Recall Curve: The primary metric for success in imbalanced credit scoring.

ROC-AUC: Measuring the separability between risk classes.

Feature Importance (Gain): Highlighting that Revolving Utilization and Number of Late Payments are the dominant predictors of financial distress.

## ⚖️ Ethical AI & Model Fairness
In line with modern financial regulations, LendRisk-Predictor includes a logic-check for:

Feature Transparency: Ensuring that model decisions are explainable via feature gain plots.

Data Integrity: Median imputation of income to ensure no borrower is unfairly penalized for missing paperwork.

## 📂 Project Structure

LendRisk-Predictor/

├── data/               # Raw and processed datasets (CSV/Excel)

├── notebooks/          # Main Jupyter/Colab Analysis

├── .gitignore          # Prevents heavy data upload to GitHub

├── README.md           # Professional documentation

└── requirements.txt    # Environment dependencies

🚀 Setup & Execution
Install Dependencies:
pip install -r requirements.txt

Download Dataset:
Place cs-training.csv in the /data folder from the Kaggle "Give Me Some Credit" competition.

Run Analysis:
Execute LendRisk_Main.ipynb to view the full training pipeline and 2x2 Evaluation Dashboard.
