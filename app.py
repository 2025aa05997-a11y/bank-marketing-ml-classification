
"""
Bank Marketing Classification - Streamlit Web Application
==========================================================
M.Tech (AIML/DSE) - Machine Learning Assignment 2

Interactive web application to demonstrate 6 classification models
trained on the UCI Bank Marketing dataset.

Features:
- Dataset upload option (CSV test data)
- Model selection dropdown
- Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing ML Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

MODEL_NAMES = [
    'Logistic Regression',
    'Decision Tree',
    'KNN',
    'Naive Bayes',
    'Random Forest',
    'XGBoost'
]

MODEL_FILES = {
    'Logistic Regression': 'model/logistic_regression_pipeline.pkl',
    'Decision Tree': 'model/decision_tree_pipeline.pkl',
    'KNN': 'model/knn_pipeline.pkl',
    'Naive Bayes': 'model/naive_bayes_pipeline.pkl',
    'Random Forest': 'model/random_forest_pipeline.pkl',
    'XGBoost': 'model/xgboost_pipeline.pkl'
}


@st.cache_resource
def load_model(model_name):
    """Load a pre-trained model pipeline from disk."""
    filepath = MODEL_FILES[model_name]
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None


@st.cache_data
def load_results():
    """Load pre-computed evaluation results."""
    if os.path.exists('model/results.pkl'):
        return joblib.load('model/results.pkl')
    return None


@st.cache_data
def load_confusion_matrices():
    """Load pre-computed confusion matrices."""
    if os.path.exists('model/confusion_matrices.pkl'):
        return joblib.load('model/confusion_matrices.pkl')
    return None


@st.cache_data
def load_column_info():
    """Load column metadata."""
    if os.path.exists('model/column_info.pkl'):
        return joblib.load('model/column_info.pkl')
    return None


@st.cache_resource
def load_label_encoder():
    """Load the target label encoder."""
    if os.path.exists('model/label_encoder.pkl'):
        return joblib.load('model/label_encoder.pkl')
    return None


@st.cache_data
def load_test_data():
    """Load the default test dataset."""
    if os.path.exists('model/test_data.csv'):
        return pd.read_csv('model/test_data.csv')
    return None


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Create a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor='white',
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def plot_metrics_comparison(results_df):
    """Create a grouped bar chart comparing all models across metrics."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(results_df.index))
    width = 0.13
    metrics = results_df.columns.tolist()
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, results_df[metric], width,
                      label=metric, color=color, alpha=0.85, edgecolor='white')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(results_df.index, rotation=15, ha='right', fontsize=10)
    ax.legend(loc='lower right', fontsize=9, ncol=3, framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute all 6 evaluation metrics."""
    metrics = {
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_true, y_pred), 4)
    }
    if y_prob is not None:
        metrics['AUC'] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        metrics['AUC'] = None
    return metrics


# ============================================================
# LOAD ALL ARTIFACTS
# ============================================================

results = load_results()
confusion_mats = load_confusion_matrices()
column_info = load_column_info()
le_target = load_label_encoder()
default_test_data = load_test_data()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ML Assignment 2")
    st.markdown("**M.Tech AIML/DSE**")
    st.markdown("---")

    # Model selection
    st.markdown("#### Select Classification Models")
    selected_model = st.selectbox(
        "Choose a model to inspect:",
        MODEL_NAMES,
        index=0
    )

    st.markdown("---")

    # CSV Upload
    st.markdown("#### Upload Test Data (CSV)")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with test data",
        type=['csv'],
        help="Upload test data CSV. Must contain the same feature columns as the training data. "
             "Include 'y' column (yes/no) for evaluation metrics."
    )

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")

    st.markdown("---")
    st.markdown(
        "**Dataset:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)"
    )
    st.markdown("**Models:** 6 classifiers")
    st.markdown("**Metrics:** Accuracy, AUC, Precision, Recall, F1, MCC")

# ============================================================
# MAIN CONTENT
# ============================================================

# Header
st.markdown('<p class="main-header">Bank Marketing - ML Classification Dashboard</p>', unsafe_allow_html=True)

# ============================================================
# TAB LAYOUT
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    " Dataset Overview",
    " Model Comparison",
    " Model Details",
    " Upload & Predict"
])

# ----------------------------------------------------------
# TAB 1: Dataset Overview
# ----------------------------------------------------------
with tab1:
    st.markdown("## Dataset Description")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Instances", "45,211")
    with col2:
        st.metric("Features", "16")
    with col3:
        st.metric("Task", "Binary Classification")
    with col4:
        st.metric("Target Variable", "y (yes/no)")

    st.markdown("---")

    st.markdown("""
    ### Problem Statement
    The data is related to **direct marketing campaigns (phone calls)** of a Portuguese banking institution.
    The **classification goal** is to predict whether a client will **subscribe to a term deposit** (variable `y`: yes/no).

    ### About the Dataset
    The Bank Marketing dataset is sourced from the **UCI Machine Learning Repository** (ID: 222).
    Marketing campaigns were based on phone calls, often requiring more than one contact to determine
    if the client would subscribe to a bank term deposit.

    **Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *A data-driven approach to predict the success
    of bank telemarketing.* Decision Support Systems.
    """)

    st.markdown("### Feature Description")

    feature_data = {
        'Feature': ['age', 'job', 'marital', 'education', 'default', 'balance',
                     'housing', 'loan', 'contact', 'day_of_week', 'month', 'duration',
                     'campaign', 'pdays', 'previous', 'poutcome'],
        'Type': ['Numeric', 'Categorical', 'Categorical', 'Categorical', 'Binary', 'Numeric',
                 'Binary', 'Binary', 'Categorical', 'Numeric', 'Categorical', 'Numeric',
                 'Numeric', 'Numeric', 'Numeric', 'Categorical'],
        'Description': [
            'Age of the client',
            'Type of job (admin, blue-collar, entrepreneur, ...)',
            'Marital status (married, divorced, single)',
            'Education level (primary, secondary, tertiary, unknown)',
            'Has credit in default? (yes/no)',
            'Average yearly balance in euros',
            'Has housing loan? (yes/no)',
            'Has personal loan? (yes/no)',
            'Contact communication type (cellular, telephone, unknown)',
            'Last contact day of the month',
            'Last contact month of year',
            'Last contact duration in seconds',
            'Number of contacts during this campaign',
            'Days since last contact from previous campaign (-1 = not contacted)',
            'Number of contacts before this campaign',
            'Outcome of previous campaign (unknown, failure, other, success)'
        ]
    }
    st.dataframe(pd.DataFrame(feature_data), use_container_width=True, hide_index=True)

    st.markdown("### Class Distribution")
    st.markdown("""
    The dataset is **imbalanced** — approximately **88% of clients did not subscribe** (class `no`)
    and only **12% subscribed** (class `yes`). This imbalance is an important consideration
    when evaluating model performance — accuracy alone can be misleading.
    """)

    if default_test_data is not None:
        st.markdown("### Sample Data (from test set)")
        st.dataframe(default_test_data.head(10), use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# TAB 2: Model Comparison
# ----------------------------------------------------------
with tab2:
    st.markdown("## Model Performance Comparison")

    if results is not None:
        results_df = pd.DataFrame(results).T
        results_df.index.name = 'Model'

        # Display comparison table
        st.markdown("### Evaluation Metrics Table")
        st.markdown("All 6 models evaluated on the same test set (20% holdout, stratified split).")

        # Styled dataframe
        styled_df = results_df.style.format("{:.4f}").background_gradient(
            cmap='YlGnBu', axis=0
        ).set_properties(**{
            'text-align': 'center',
            'font-size': '14px'
        })
        st.dataframe(styled_df, use_container_width=True)

        # Best model highlights
        st.markdown("### Best Model per Metric")
        best_cols = st.columns(6)
        for i, metric in enumerate(results_df.columns):
            best_model = results_df[metric].idxmax()
            best_val = results_df[metric].max()
            with best_cols[i]:
                st.metric(
                    label=metric,
                    value=f"{best_val:.4f}",
                    help=f"Best: {best_model}"
                )
                st.caption(f"**{best_model}**")

        # Comparison chart
        st.markdown("### Visual Comparison")
        fig = plot_metrics_comparison(results_df)
        st.pyplot(fig)
        plt.close()

        # Radar / summary
        st.markdown("### Key Observations")
        st.markdown("""
        - **Ensemble methods** (Random Forest, XGBoost) generally outperform single models
        - **Class imbalance** significantly affects Recall and F1 scores
        - **AUC** is a more reliable metric than accuracy for this imbalanced dataset
        - **MCC** provides a balanced measure considering all quadrants of the confusion matrix
        """)
    else:
        st.warning("No pre-computed results found. Please run `model/train_models.py` first.")

# ----------------------------------------------------------
# TAB 3: Model Details
# ----------------------------------------------------------
with tab3:
    st.markdown(f"## Model Details: {selected_model}")

    if results is not None and selected_model in results:
        metrics = results[selected_model]

        # Display metrics in cards
        st.markdown("### Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        metric_items = list(metrics.items())

        for i, (metric, value) in enumerate(metric_items):
            col = [col1, col2, col3][i % 3]
            with col:
                # Color-code: green for high, red for low
                delta_color = "normal"
                st.metric(label=metric, value=f"{value:.4f}")

        # Confusion Matrix
        st.markdown("### Confusion Matrix")
        if confusion_mats is not None and selected_model in confusion_mats:
            cm = np.array(confusion_mats[selected_model])
            class_names = column_info['target_classes'] if column_info else ['no', 'yes']
            fig = plot_confusion_matrix(cm, class_names, title=f"{selected_model} - Confusion Matrix")
            st.pyplot(fig)
            plt.close()

            # Confusion matrix interpretation
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
            **Confusion Matrix Breakdown:**
            | | Predicted No | Predicted Yes |
            |---|---|---|
            | **Actual No** | {tn} (TN) | {fp} (FP) |
            | **Actual Yes** | {fn} (FN) | {tp} (TP) |

            - **True Positives (TP):** {tp} — Correctly predicted subscriptions
            - **True Negatives (TN):** {tn} — Correctly predicted non-subscriptions
            - **False Positives (FP):** {fp} — Incorrectly predicted as subscribed (Type I error)
            - **False Negatives (FN):** {fn} — Missed actual subscriptions (Type II error)
            """)

        # Classification Report
        st.markdown("### Classification Report")
        if default_test_data is not None and le_target is not None:
            pipeline = load_model(selected_model)
            if pipeline is not None:
                X_test_default = default_test_data.drop(columns=['y'], errors='ignore')
                y_test_default = le_target.transform(default_test_data['y'].values)
                y_pred_default = pipeline.predict(X_test_default)
                report = classification_report(
                    y_test_default, y_pred_default,
                    target_names=le_target.classes_,
                    output_dict=True
                )
                report_df = pd.DataFrame(report).T
                st.dataframe(
                    report_df.style.format("{:.4f}", subset=pd.IndexSlice[:, ['precision', 'recall', 'f1-score']]),
                    use_container_width=True
                )

        # Model-specific observations
        st.markdown("### Model Observation")
        observations = {
            'Logistic Regression': """
            **Logistic Regression** serves as a strong baseline model. It assumes a linear decision boundary
            between classes. On this dataset, it achieves reasonable accuracy but may struggle with the
            minority class (subscribers) due to class imbalance. The model benefits from feature scaling
            (StandardScaler) and handles the one-hot encoded categorical features well. Its strength lies
            in interpretability — the learned coefficients directly indicate feature importance and direction
            of influence on the prediction.
            """,
            'Decision Tree': """
            **Decision Tree** captures non-linear relationships and feature interactions that linear models miss.
            With `max_depth=10` regularization, it avoids severe overfitting while still learning complex patterns.
            Decision trees are inherently interpretable and can be visualized. However, they tend to be
            greedy learners and may not find the globally optimal split. The model's performance on this
            imbalanced dataset shows it can identify some minority class patterns but is sensitive to
            the majority class dominance.
            """,
            'KNN': """
            **K-Nearest Neighbors (K=7, distance-weighted)** is an instance-based learner that makes predictions
            based on the similarity of test instances to training examples. Feature scaling is critical for KNN
            as it relies on distance calculations. The distance-weighted voting gives closer neighbors more
            influence. On this dataset, KNN faces challenges due to the high dimensionality after one-hot
            encoding (curse of dimensionality) and the class imbalance, which can bias the neighborhood
            toward the majority class.
            """,
            'Naive Bayes': """
            **Gaussian Naive Bayes** assumes features are independent given the class label — a strong but
            often violated assumption (feature independence). Despite this, it performs surprisingly well as a
            fast probabilistic classifier. On this dataset, the conditional independence assumption may not
            fully hold (e.g., `education` and `job` are correlated), which can limit its discriminative power.
            Its strength is computational efficiency and good calibration of probability estimates, making it
            useful as a baseline probabilistic model.
            """,
            'Random Forest': """
            **Random Forest (150 trees)** is an ensemble of decorrelated decision trees trained on bootstrap
            samples with random feature subsets. This reduces variance compared to a single decision tree and
            typically yields superior generalization. On the Bank Marketing dataset, Random Forest handles the
            mix of categorical and numerical features effectively and is robust to feature scaling. The ensemble
            averaging smooths out individual tree biases, leading to better minority class detection. Feature
            importance from the forest reveals `duration`, `balance`, and `age` as top predictors.
            """,
            'XGBoost': """
            **XGBoost (150 trees, lr=0.1)** is a gradient boosting framework that builds trees sequentially,
            with each tree correcting the errors of the previous ensemble. It typically achieves the best
            performance on structured/tabular data. On this dataset, XGBoost's regularization (L1/L2), learning
            rate scheduling, and built-in handling of imbalanced classes (via scale_pos_weight) make it well-suited.
            The sequential error correction mechanism means it can focus on hard-to-classify minority class
            examples in later boosting rounds, improving Recall and F1.
            """
        }
        st.markdown(observations.get(selected_model, "No observation available."))

    else:
        st.warning("Model results not available. Please run `model/train_models.py` first.")

# ----------------------------------------------------------
# TAB 4: Upload & Predict
# ----------------------------------------------------------
with tab4:
    st.markdown("## Upload Test Data & Predict")

    if uploaded_file is not None:
        try:
            # Read uploaded CSV
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {uploaded_df.shape[0]} rows and {uploaded_df.shape[1]} columns.")

            st.markdown("### Uploaded Data Preview")
            st.dataframe(uploaded_df.head(10), use_container_width=True, hide_index=True)

            # Check if target column exists
            has_target = 'y' in uploaded_df.columns

            if has_target:
                X_upload = uploaded_df.drop(columns=['y'])
                y_upload_raw = uploaded_df['y'].values
                if le_target is not None:
                    y_upload = le_target.transform(y_upload_raw)
                else:
                    y_upload = np.where(y_upload_raw == 'yes', 1, 0)
            else:
                X_upload = uploaded_df
                y_upload = None

            # Validate columns
            if column_info is not None:
                expected_features = column_info['feature_names']
                upload_features = X_upload.columns.tolist()
                missing_cols = set(expected_features) - set(upload_features)
                extra_cols = set(upload_features) - set(expected_features)

                if missing_cols:
                    st.warning(f"Missing columns in uploaded data: {missing_cols}")
                if extra_cols:
                    st.info(f"Extra columns (will be ignored): {extra_cols}")
                    X_upload = X_upload[[c for c in expected_features if c in X_upload.columns]]

            # Load selected model and predict
            pipeline = load_model(selected_model)
            if pipeline is not None:
                st.markdown(f"### Predictions using **{selected_model}**")

                y_pred = pipeline.predict(X_upload)
                y_prob = pipeline.predict_proba(X_upload)[:, 1]

                # Show predictions
                pred_df = uploaded_df.copy()
                pred_df['Predicted'] = le_target.inverse_transform(y_pred) if le_target else y_pred
                pred_df['Probability (Yes)'] = np.round(y_prob, 4)
                st.dataframe(pred_df.head(20), use_container_width=True, hide_index=True)

                # If target exists, compute metrics
                if has_target and y_upload is not None:
                    st.markdown("### Evaluation on Uploaded Data")
                    upload_metrics = compute_metrics(y_upload, y_pred, y_prob)

                    col1, col2, col3 = st.columns(3)
                    metric_items = list(upload_metrics.items())
                    for i, (metric, value) in enumerate(metric_items):
                        if value is not None:
                            col = [col1, col2, col3][i % 3]
                            with col:
                                st.metric(label=metric, value=f"{value:.4f}")

                    # Confusion Matrix for uploaded data
                    st.markdown("### Confusion Matrix (Uploaded Data)")
                    cm_upload = confusion_matrix(y_upload, y_pred)
                    class_names = le_target.classes_ if le_target else ['no', 'yes']
                    fig = plot_confusion_matrix(
                        cm_upload, class_names,
                        title=f"{selected_model} - Uploaded Data Confusion Matrix"
                    )
                    st.pyplot(fig)
                    plt.close()

                    # Classification Report
                    st.markdown("### Classification Report (Uploaded Data)")
                    report = classification_report(
                        y_upload, y_pred,
                        target_names=class_names,
                        output_dict=True
                    )
                    report_df = pd.DataFrame(report).T
                    st.dataframe(report_df, use_container_width=True)

                else:
                    st.info(
                        "Target column 'y' not found in uploaded data. "
                        "Showing predictions only. Add a 'y' column (yes/no) to see evaluation metrics."
                    )

                # Download predictions
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"Model '{selected_model}' not found. Please run `model/train_models.py` first.")

        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            st.info("Please ensure the CSV file has the correct format with the expected feature columns.")

    else:
        st.info("Please upload a CSV file using the sidebar to make predictions.")
        st.markdown("""
        ### Instructions
        1. Use the **sidebar** to select a model and upload a CSV test file
        2. The CSV should contain the same feature columns as the training data
        3. Optionally include a `y` column (yes/no) to see evaluation metrics
        4. A sample test file (`model/test_data.csv`) is included in the repository

        ### Expected CSV Columns
        `age, job, marital, education, default, balance, housing, loan, contact, day_of_week, month, duration, campaign, pdays, previous, poutcome`

        Optionally include the target column: `y` (yes/no)
        """)

        if default_test_data is not None:
            st.markdown("### Download Sample Test Data")
            csv = default_test_data.to_csv(index=False)
            st.download_button(
                label="Download test_data.csv",
                data=csv,
                file_name="test_data.csv",
                mime="text/csv"
            )


