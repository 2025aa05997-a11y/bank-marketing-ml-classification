"""
Bank Marketing Classification - Model Training Script
=====================================================
M.Tech (AIML/DSE) - Machine Learning Assignment 2

This script:
1. Fetches the Bank Marketing dataset from UCI ML Repository
2. Performs data preprocessing (encoding, scaling)
3. Trains 6 classification models
4. Evaluates each model on 6 metrics
5. Saves trained pipelines, results, and test data for the Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ============================================================
# 1. FETCH AND EXPLORE THE DATASET
# ============================================================

print("=" * 70)
print("STEP 1: Fetching Bank Marketing Dataset from UCI Repository (id=222)")
print("=" * 70)

bank_marketing = fetch_ucirepo(id=222)

X = bank_marketing.data.features
y = bank_marketing.data.targets

print(f"\nDataset Shape: {X.shape[0]} instances, {X.shape[1]} features")
print(f"Target Distribution:\n{y.value_counts().to_string()}")
print(f"\nFeature Columns: {X.columns.tolist()}")
print(f"\nData Types:\n{X.dtypes.to_string()}")

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Data Preprocessing")
print("=" * 70)

# Encode target variable: 'yes' -> 1, 'no' -> 0
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y.values.ravel())

print(f"\nTarget Encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
print(f"Class Distribution after encoding:")
unique, counts = np.unique(y_encoded, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls} ({le_target.inverse_transform([cls])[0]}): {cnt} ({cnt/len(y_encoded)*100:.1f}%)")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

print(f"\nCategorical Features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical Features ({len(numerical_cols)}): {numerical_cols}")

# Handle any missing values
total_missing = X.isnull().sum().sum()
if total_missing > 0:
    print(f"\nWarning: Found {total_missing} missing values. Filling with mode/median.")
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].median())
    print(f"After filling: {X.isnull().sum().sum()} missing values remain.")
else:
    print("\nNo missing values found in the dataset.")

# Create preprocessing pipeline using ColumnTransformer
# - StandardScaler for numerical features (important for KNN, Logistic Regression)
# - OneHotEncoder for categorical features (creates binary dummy variables)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: Train-Test Split (80-20, Stratified)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Save test data as CSV for Streamlit upload feature
test_data_with_target = X_test.copy()
test_data_with_target['y'] = le_target.inverse_transform(y_test)
test_data_with_target.to_csv('test_data.csv', index=False)
print(f"\nTest data saved to test_data.csv ({test_data_with_target.shape[0]} rows)")

# ============================================================
# 4. MODEL DEFINITION
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: Defining 6 Classification Models")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        C=1.0
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='minkowski',
        p=2
    ),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=150,
        random_state=42,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1
    )
}

for name, model in models.items():
    print(f"\n  {name}: {model.__class__.__name__}")
    params = model.get_params()
    key_params = {k: v for k, v in params.items() if k in [
        'max_iter', 'C', 'solver', 'max_depth', 'min_samples_split',
        'n_neighbors', 'weights', 'n_estimators', 'learning_rate', 'eval_metric'
    ]}
    if key_params:
        print(f"    Key Parameters: {key_params}")

# ============================================================
# 5. TRAINING AND EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: Training and Evaluating All Models")
print("=" * 70)

results = {}
confusion_matrices = {}

for name, model in models.items():
    print(f"\n{'-' * 50}")
    print(f"Training: {name}")
    print(f"{'-' * 50}")

    # Create pipeline: preprocessor + classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate all 6 metrics
    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_prob), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1': round(f1_score(y_test, y_pred), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }

    results[name] = metrics
    confusion_matrices[name] = confusion_matrix(y_test, y_pred).tolist()

    # Print metrics
    for metric, value in metrics.items():
        print(f"  {metric:>12s}: {value:.4f}")

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>15s} Predicted No  Predicted Yes")
    print(f"  {'Actual No':>15s}    {cm[0][0]:>6d}        {cm[0][1]:>6d}")
    print(f"  {'Actual Yes':>15s}    {cm[1][0]:>6d}        {cm[1][1]:>6d}")

    # Print classification report
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=le_target.classes_)
    for line in report.split('\n'):
        print(f"  {line}")

    # Save the trained pipeline
    model_filename = f"{name.lower().replace(' ', '_')}_pipeline.pkl"
    joblib.dump(pipeline, model_filename)
    print(f"\n  Model saved to: {model_filename}")

# ============================================================
# 6. RESULTS SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: Results Summary - Comparison Table")
print("=" * 70)

# Create comparison DataFrame
results_df = pd.DataFrame(results).T
results_df.index.name = 'Model'
print(f"\n{results_df.to_string()}")

# Find best model for each metric
print(f"\n{'-' * 50}")
print("Best Model per Metric:")
print(f"{'-' * 50}")
for metric in results_df.columns:
    best_model = results_df[metric].idxmax()
    best_value = results_df[metric].max()
    print(f"  {metric:>12s}: {best_model} ({best_value:.4f})")

# ============================================================
# 7. SAVE ALL ARTIFACTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: Saving Artifacts")
print("=" * 70)

# Save results dictionary
joblib.dump(results, 'results.pkl')
print("  Saved: results.pkl (evaluation metrics)")

# Save confusion matrices
joblib.dump(confusion_matrices, 'confusion_matrices.pkl')
print("  Saved: confusion_matrices.pkl")

# Save label encoder
joblib.dump(le_target, 'label_encoder.pkl')
print("  Saved: label_encoder.pkl")

# Save column information for the Streamlit app
column_info = {
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'feature_names': X.columns.tolist(),
    'target_classes': le_target.classes_.tolist()
}
joblib.dump(column_info, 'column_info.pkl')
print("  Saved: column_info.pkl")

import os
os.makedirs('model', exist_ok=True)
# Save results as CSV for easy viewing
results_df.to_csv('model/results_summary.csv')
print("  Saved: model/results_summary.csv")

print("\n" + "=" * 70)
print("ALL DONE! Models trained, evaluated, and saved successfully.")
print("=" * 70)
print("\nFiles created in model/ directory:")
for f in sorted(os.listdir('model')):
    size = os.path.getsize(f'model/{f}')
    print(f"  {f:45s} ({size/1024:.1f} KB)")

print("\nNext: Run 'streamlit run app.py' to launch the web application.")
