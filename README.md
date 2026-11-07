# M.Tech ML Assignment 2 - Obesity Level Classification & Deployment

## a. Problem statement
The objective of this assignment is to build, evaluate, and deploy multiple machine learning classification models to predict obesity levels (`NObeyesdad`) based on physical condition, dietary habits, and lifestyle features.

## b. Dataset description
- **Dataset Name:** Obesity Risk Prediction Dataset (UCI/Kaggle)
- **Features (16):** Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS
- **Target Variable:** `NObeyesdad` (Multi-class obesity levels)
- **Instance Size:** 2,111 records (Satisfies minimum requirement of 500)
- **Feature Size:** 16 features (Satisfies minimum requirement of 12)

## c. GitHub Repository Link
[Insert your GitHub Repository URL here]

## d. Models used
### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8676 | 0.9855 | 0.8676 | 0.8676 | 0.8656 | 0.846 |
| Decision Tree | 0.9173 | 0.9519 | 0.9206 | 0.9173 | 0.9182 | 0.9036 |
| kNN | 0.8061 | 0.948 | 0.8138 | 0.8061 | 0.7987 | 0.7772 |
| Naive Bayes | 0.6028 | 0.9058 | 0.6452 | 0.6028 | 0.5792 | 0.5474 |
| Support Vector Machine | 0.87 | 0.9876 | 0.8736 | 0.87 | 0.871 | 0.8484 |

### Observations about model performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performs well with ~86.8% accuracy, effectively capturing primary linear patterns after standardization. |
| **Decision Tree** | Top performer (~91.7% accuracy and 0.9036 MCC), capturing complex non-linear feature splits across physical and dietary metrics. |
| **kNN** | Solid multi-class baseline (~80.6% accuracy), but performance is slightly reduced by feature dimensionality. |
| **Naive Bayes** | Lowest accuracy (~60.3%) due to strong independent feature assumptions which are violated by correlated biometric attributes. |
| **Support Vector Machine** | Strong continuous boundary modeling (~87.0% accuracy and highest AUC of 0.9876) using RBF kernel. |
| **Overall Winner for your dataset?** | **Decision Tree** achieved the highest overall accuracy, F1 score, and MCC score for multi-class obesity classification. |
