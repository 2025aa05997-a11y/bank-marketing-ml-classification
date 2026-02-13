# Bank Marketing - ML Classification

## Problem Statement

The goal of this project is to predict whether a client of a Portuguese banking institution will **subscribe to a term deposit** based on data from direct marketing campaigns (phone calls). This is a **binary classification problem** where the target variable `y` takes values `yes` (subscribed) or `no` (did not subscribe).

We implement and compare **6 machine learning classification models** on this dataset, evaluating each using 6 standard metrics: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

---

## Dataset Description

| Property | Value |
|---|---|
| **Source** | [UCI Machine Learning Repository - Bank Marketing (ID: 222)](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Instances** | 45,211 |
| **Features** | 16 |
| **Target Variable** | `y` — Has the client subscribed a term deposit? (yes/no) |
| **Task** | Binary Classification |
| **Class Distribution** | ~88% No, ~12% Yes (imbalanced) |
| **Missing Values** | None |

### Feature Details

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `age` | Numeric | Age of the client |
| 2 | `job` | Categorical | Type of job (admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown) |
| 3 | `marital` | Categorical | Marital status (married, divorced, single) |
| 4 | `education` | Categorical | Education level (primary, secondary, tertiary, unknown) |
| 5 | `default` | Binary | Has credit in default? (yes/no) |
| 6 | `balance` | Numeric | Average yearly balance in euros |
| 7 | `housing` | Binary | Has housing loan? (yes/no) |
| 8 | `loan` | Binary | Has personal loan? (yes/no) |
| 9 | `contact` | Categorical | Contact communication type (cellular, telephone, unknown) |
| 10 | `day_of_week` | Numeric | Last contact day of the month |
| 11 | `month` | Categorical | Last contact month of year |
| 12 | `duration` | Numeric | Last contact duration in seconds |
| 13 | `campaign` | Numeric | Number of contacts during this campaign |
| 14 | `pdays` | Numeric | Days since last contact from previous campaign (-1 = not contacted) |
| 15 | `previous` | Numeric | Number of contacts before this campaign |
| 16 | `poutcome` | Categorical | Outcome of previous campaign (unknown, failure, other, success) |

**Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems. DOI: 10.24432/C5K306

---

## Project Structure

```
project-folder/
│── app.py                  # Streamlit web application
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
│── model/
│   ├── train_models.py     # Model training and evaluation script
│   ├── test_data.csv       # Test dataset for upload
│   ├── results.pkl         # Pre-computed evaluation metrics
│   ├── confusion_matrices.pkl
│   ├── label_encoder.pkl
│   ├── column_info.pkl
│   ├── results_summary.csv
│   ├── logistic_regression_pipeline.pkl
│   ├── decision_tree_pipeline.pkl
│   ├── knn_pipeline.pkl
│   ├── naive_bayes_pipeline.pkl
│   ├── random_forest_pipeline.pkl
│   └── xgboost_pipeline.pkl
```

---

## Models Used

Six classification models were implemented, each wrapped in an sklearn `Pipeline` with consistent preprocessing (StandardScaler for numerical features, OneHotEncoder for categorical features):

1. **Logistic Regression** — Linear model, `max_iter=1000`, `solver=lbfgs`
2. **Decision Tree Classifier** — `max_depth=10`, `min_samples_split=5`
3. **K-Nearest Neighbors (KNN)** — `n_neighbors=7`, `weights=distance`
4. **Gaussian Naive Bayes** — Probabilistic classifier assuming feature independence
5. **Random Forest (Ensemble)** — `n_estimators=150`, `max_depth=15`
6. **XGBoost (Ensemble)** — `n_estimators=150`, `learning_rate=0.1`, `max_depth=6`

### Preprocessing Pipeline
- **Numerical features:** StandardScaler (zero mean, unit variance)
- **Categorical features:** OneHotEncoder (with `drop='first'` to avoid multicollinearity)
- **Train-Test Split:** 80-20, stratified on the target variable, `random_state=42`

---

## Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8990 | 0.9006 | 0.6325 | 0.3270 | 0.4312 | 0.4070 |
| Decision Tree | 0.8997 | 0.8476 | 0.6197 | 0.3696 | 0.4630 | 0.4283 |
| KNN | 0.8964 | 0.8483 | 0.6118 | 0.3129 | 0.4140 | 0.3884 |
| Naive Bayes | 0.8693 | 0.8083 | 0.4445 | 0.4698 | 0.4568 | 0.3828 |
| Random Forest (Ensemble) | 0.9041 | 0.9183 | 0.6826 | 0.3374 | 0.4516 | 0.4360 |
| XGBoost (Ensemble) | **0.9070** | **0.9264** | 0.6445 | 0.4575 | **0.5351** | **0.4939** |

**Best model per metric:**
- **Accuracy:** XGBoost (0.9070)
- **AUC:** XGBoost (0.9264)
- **Precision:** Random Forest (0.6826)
- **Recall:** Naive Bayes (0.4698)
- **F1 Score:** XGBoost (0.5351)
- **MCC:** XGBoost (0.4939)

---

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Achieves 89.90% accuracy and a strong AUC of 0.9006, demonstrating that even a linear model can separate the classes well in probability space. However, its Recall is only 0.3270 — it misses ~67% of actual subscribers — because the linear decision boundary struggles with the complex, non-linear patterns in the data. Precision (0.6325) is moderate, meaning about 63% of its positive predictions are correct. The low MCC (0.4070) confirms limited balanced performance. Strength: highly interpretable; coefficients reveal which features (e.g., `duration`, `poutcome=success`) most influence subscription probability. |
| **Decision Tree** | Slightly outperforms Logistic Regression in Recall (0.3696 vs 0.3270) by capturing non-linear decision rules and feature interactions. The `max_depth=10` constraint prevents overfitting but limits the tree's ability to model fine-grained minority class patterns. AUC drops to 0.8476 (lower than Logistic Regression), indicating the tree's probability estimates are less well-calibrated. F1 of 0.4630 is a modest improvement. The greedy splitting nature means it may find locally optimal but globally suboptimal boundaries. Key advantage: produces interpretable if-then rules that domain experts can validate. |
| **KNN** | Shows the weakest overall performance: lowest Accuracy (0.8964), lowest Recall (0.3129), and lowest F1 (0.4140). After one-hot encoding of 9 categorical features, the feature space expands significantly (~30+ dimensions), causing the **curse of dimensionality** — distances become less meaningful in high dimensions. Despite distance-weighted voting (K=7), the 88:12 class imbalance means most neighborhoods are dominated by the majority class. Additionally, KNN's model size (11 MB) is the second largest since it stores all training instances. Not recommended for this dataset given the high dimensionality and imbalance. |
| **Naive Bayes** | Achieves the **highest Recall** among all models (0.4698) — it identifies ~47% of actual subscribers, trading precision (0.4445) for better minority class detection. This makes it the most aggressive in predicting the positive class, resulting in the lowest Accuracy (0.8693) due to more false positives (621 FP). The conditional independence assumption is violated (e.g., `job` and `education` are correlated), limiting its AUC to 0.8083 (lowest). However, for a campaign where missing potential subscribers is costly, its higher Recall may be strategically valuable despite lower overall accuracy. |
| **Random Forest (Ensemble)** | Achieves the **highest Precision** (0.6826) — when it predicts a client will subscribe, it is correct 68% of the time. This comes from the variance-reducing effect of averaging 150 decorrelated trees on bootstrap samples. AUC is strong at 0.9183 (second best), confirming excellent class separation. However, Recall (0.3374) is moderate — it still misses ~66% of actual subscribers, being conservative in positive predictions. The ensemble effectively handles the mixed feature types and the imbalance better than any single model. Feature importance typically shows `duration`, `balance`, and `poutcome` as top predictors. |
| **XGBoost (Ensemble)** | **Best overall performer** — achieves the highest Accuracy (0.9070), AUC (0.9264), F1 (0.5351), and MCC (0.4939). Its sequential boosting mechanism allows later trees to focus on correcting errors from the minority class, yielding significantly better Recall (0.4575) than all models except Naive Bayes, while maintaining good Precision (0.6445). The F1 of 0.5351 represents the best precision-recall balance. L1/L2 regularization and a learning rate of 0.1 prevent overfitting. XGBoost's gradient boosting approach is well-suited for tabular data with mixed feature types and class imbalance — it is the recommended model for deployment. |

### Overall Analysis

1. **Class Imbalance Impact:** The dataset has 88.3% negative class (no) and only 11.7% positive class (yes). All models achieve ~87-91% accuracy, but a naive classifier predicting "no" always would get 88.3%. This makes Accuracy alone misleading. AUC, F1, and MCC are more informative — and here XGBoost leads clearly (AUC: 0.9264, F1: 0.5351, MCC: 0.4939).

2. **Ensemble Superiority:** Random Forest and XGBoost outperform all single models. XGBoost achieves the best scores on 4 of 6 metrics. Random Forest leads in Precision (0.6826). The improvement from Decision Tree to Random Forest demonstrates how bagging reduces variance, while XGBoost's boosting corrects residual errors sequentially.

3. **Precision-Recall Trade-off is Clearly Visible:** Naive Bayes has the highest Recall (0.4698) but the lowest Precision (0.4445). Random Forest has the highest Precision (0.6826) but lower Recall (0.3374). XGBoost achieves the best balance (F1: 0.5351). For a bank's marketing campaign, the choice depends on cost: if false positives (wasted calls) are cheap, maximize Recall (Naive Bayes); if campaign resources are limited, maximize Precision (Random Forest); for overall balance, use XGBoost.

4. **Feature Importance:** Across tree-based models, `duration` (call duration) is the most predictive feature, followed by `balance`, `poutcome` (previous campaign outcome), and `month`. **Important caveat:** `duration` is known only after the call ends, making it unavailable for prospective prediction. In a real deployment, one should retrain without `duration` for a truly prospective model.

5. **MCC as the Most Reliable Metric:** Matthews Correlation Coefficient considers all four confusion matrix quadrants (TP, TN, FP, FN) and ranges from -1 to +1. XGBoost's MCC of 0.4939 is the highest, confirming it is genuinely the best at distinguishing both classes. The relatively modest MCC values across all models (0.38-0.49) reflect the inherent difficulty of predicting a rare event (11.7%) from marketing call data.

6. **KNN Limitations in High Dimensions:** KNN shows the weakest performance overall, suffering from the curse of dimensionality after one-hot encoding expands the feature space. It also has the largest model file (11 MB) since it stores all training instances. For datasets with many categorical features, distance-based methods are generally less effective than tree-based approaches.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
cd model
python train_models.py
```
This will fetch the dataset, train all 6 models, evaluate them, and save all artifacts in the `model/` directory.

### 3. Run Streamlit App
```bash
streamlit run app.py
```

### 4. Deploy on Streamlit Community Cloud
1. Push this repository to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New App" → Select repository → Choose `app.py` → Deploy

---

## Streamlit App Features

- **Dataset Upload (CSV):** Upload test data to get predictions and evaluation metrics
- **Model Selection Dropdown:** Choose from 6 trained classification models
- **Evaluation Metrics Display:** Accuracy, AUC, Precision, Recall, F1, MCC for each model
- **Confusion Matrix:** Visual heatmap with detailed breakdown (TP, TN, FP, FN)
- **Classification Report:** Per-class precision, recall, and F1 scores
- **Model Comparison:** Side-by-side comparison table and bar chart of all models
- **Download Predictions:** Export predictions as CSV

---

## Tech Stack

- **Python 3.9+**
- **scikit-learn** — ML models, preprocessing, metrics
- **XGBoost** — Gradient boosting classifier
- **Streamlit** — Interactive web application
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualization
- **ucimlrepo** — Dataset fetching from UCI repository
- **joblib** — Model serialization

---

*M.Tech (AIML/DSE) — Machine Learning Assignment 2*
