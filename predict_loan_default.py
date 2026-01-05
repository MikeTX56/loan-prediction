"""
Loan Default Prediction Model
Predicts whether a loan will be good (repaid) or bad (defaulted)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Data directory
DATA_DIR = 'data-science-nigeria-challenge-1-loan-default-prediction20250307-26022-im3qg9'


def load_and_prepare_data(data_type='train'):
    """Load and merge datasets"""
    print(f"Loading {data_type} data...")
    
    # Load datasets
    perf = pd.read_csv(f'{DATA_DIR}/{data_type}perf.csv')
    demo = pd.read_csv(f'{DATA_DIR}/{data_type}demographics.csv')
    prev = pd.read_csv(f'{DATA_DIR}/{data_type}prevloans.csv')
    
    # Merge performance with demographics
    data = perf.merge(demo, on='customerid', how='left')
    
    # Aggregate previous loans data
    if len(prev) > 0:
        prev_agg = aggregate_previous_loans(prev)
        data = data.merge(prev_agg, on='customerid', how='left')
    
    return data


def aggregate_previous_loans(prev_loans):
    """Aggregate previous loan history features"""
    # Convert date columns to datetime
    date_cols = ['approveddate', 'creationdate', 'closeddate', 'firstduedate', 'firstrepaiddate']
    for col in date_cols:
        if col in prev_loans.columns:
            prev_loans[col] = pd.to_datetime(prev_loans[col], errors='coerce')
    
    # Calculate days to close loan
    prev_loans['days_to_close'] = (prev_loans['closeddate'] - prev_loans['approveddate']).dt.days
    
    # Calculate if paid early (before or on due date)
    prev_loans['paid_early'] = (prev_loans['firstrepaiddate'] <= prev_loans['firstduedate']).astype(int)
    
    # Calculate days early/late
    prev_loans['days_diff_payment'] = (prev_loans['firstrepaiddate'] - prev_loans['firstduedate']).dt.days
    
    # Aggregate by customer
    agg_dict = {
        'systemloanid': 'count',  # number of previous loans
        'loanamount': ['mean', 'sum', 'max'],
        'totaldue': ['mean', 'sum'],
        'termdays': 'mean',
        'days_to_close': ['mean', 'min', 'max'],
        'paid_early': ['sum', 'mean'],
        'days_diff_payment': ['mean', 'min', 'max']
    }
    
    prev_agg = prev_loans.groupby('customerid').agg(agg_dict)
    prev_agg.columns = ['prev_' + '_'.join(col).strip() for col in prev_agg.columns.values]
    prev_agg = prev_agg.reset_index()
    
    return prev_agg


def engineer_features(data):
    """Feature engineering"""
    print("Engineering features...")
    
    # Convert dates to datetime
    data['approveddate'] = pd.to_datetime(data['approveddate'], errors='coerce')
    data['creationdate'] = pd.to_datetime(data['creationdate'], errors='coerce')
    data['birthdate'] = pd.to_datetime(data['birthdate'], errors='coerce')
    
    # Age at loan application
    data['age'] = (data['approveddate'] - data['birthdate']).dt.days / 365.25
    
    # Time from creation to approval (in hours)
    data['creation_to_approval_hours'] = (data['approveddate'] - data['creationdate']).dt.total_seconds() / 3600
    
    # Loan to term ratio
    data['loan_per_day'] = data['loanamount'] / data['termdays']
    
    # Interest rate
    data['interest_rate'] = (data['totaldue'] - data['loanamount']) / data['loanamount']
    
    # Drop date columns
    date_cols = ['approveddate', 'creationdate', 'birthdate']
    data = data.drop(columns=date_cols, errors='ignore')
    
    # Fill missing values for previous loan features with 0 (no previous loans)
    prev_cols = [col for col in data.columns if col.startswith('prev_')]
    for col in prev_cols:
        data[col] = data[col].fillna(0)
    
    # Handle categorical variables
    categorical_cols = ['bank_account_type', 'bank_name_clients', 'employment_status_clients', 
                       'level_of_education_clients']
    
    # Fill missing values in categorical columns
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')
    
    # Label encode categorical variables
    le_dict = {}
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            le_dict[col] = le
    
    # Fill remaining missing values
    data['bank_branch_clients'] = data['bank_branch_clients'].fillna('Unknown')
    data = data.drop(columns=['bank_branch_clients'], errors='ignore')  # Too many missing values
    
    # Fill referredby with 0 (not referred)
    if 'referredby' in data.columns:
        data['referredby'] = data['referredby'].notna().astype(int)
    
    return data, le_dict


def prepare_features(data, target_col='good_bad_flag'):
    """Prepare features for modeling"""
    # Identify feature columns (exclude ID and target)
    exclude_cols = ['customerid', 'systemloanid', target_col]
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols]
    
    # Handle any remaining missing values
    X = X.fillna(X.mean(numeric_only=True))
    
    if target_col in data.columns:
        # Convert target to binary (Good=1, Bad=0)
        y = (data[target_col] == 'Good').astype(int)
        return X, y, feature_cols
    else:
        return X, None, feature_cols


def train_model(X_train, y_train):
    """Train XGBoost classifier"""
    print("Training model...")
    
    # Initialize model with parameters to handle imbalanced data
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=3.6  # Ratio of negative to positive class
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def main():
    """Main execution function"""
    print("=" * 50)
    print("Loan Default Prediction Model")
    print("=" * 50)
    
    # Load training data
    train_data = load_and_prepare_data('train')
    
    # Engineer features
    train_data, le_dict = engineer_features(train_data)
    
    # Prepare features
    X, y, feature_cols = prepare_features(train_data)
    
    print(f"\nTraining data shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Target distribution: Good={sum(y)}, Bad={len(y)-sum(y)}")
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Validate
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Error Rate: {(1-val_accuracy)*100:.2f}%")
    
    # Train on full data
    print("\nTraining on full dataset...")
    model = train_model(X, y)
    
    # Load test data
    print("\n" + "=" * 50)
    print("Generating Predictions on Test Data")
    print("=" * 50)
    
    test_data = load_and_prepare_data('test')
    
    # Engineer features (same transformations)
    test_data, _ = engineer_features(test_data)
    
    # Prepare features
    X_test, _, _ = prepare_features(test_data, target_col=None)
    
    # Ensure test features match training features
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_cols]
    
    # Handle any remaining missing values
    X_test = X_test.fillna(X_test.mean(numeric_only=True))
    
    print(f"Test data shape: {X_test.shape}")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'customerid': test_data['customerid'],
        'Good_Bad_flag': predictions
    })
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print(f"\nSubmission file created: submission.csv")
    print(f"Total predictions: {len(submission)}")
    print(f"Predicted Good: {sum(predictions)}, Predicted Bad: {len(predictions)-sum(predictions)}")
    
    # Display sample
    print("\nSample predictions:")
    print(submission.head(10))
    
    print("\n" + "=" * 50)
    print("Prediction Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
