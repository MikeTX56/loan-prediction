# Loan Default Prediction

This project predicts whether a loan will be good (repaid) or bad (defaulted) for SuperLender, a digital lending company.

## Problem Statement

The goal is to predict if a loan was good or bad, i.e., accurately predict binary outcome variable, where Good is 1 and Bad is 0.

The assessment is based on two main risk drivers:
1. **Willingness to pay** - Customer's intention to repay
2. **Ability to pay** - Customer's financial capacity to repay

## Data

The dataset includes:
- **Demographics**: Customer information (birthdate, bank details, employment, education)
- **Performance**: Current loan details (loan amount, term, dates)
- **Previous Loans**: Historical loan performance for repeat customers

## Solution

The solution uses machine learning to predict loan outcomes:
1. **Data preprocessing and feature engineering**:
   - Merge demographics, performance, and previous loan history
   - Calculate age from birthdate
   - Aggregate previous loan metrics (count, amounts, payment behavior)
   - Create derived features (loan per day, interest rate, payment patterns)
2. **Training a classification model**:
   - XGBoost classifier with parameters tuned for imbalanced data
   - Scale_pos_weight to handle class imbalance (more Good loans than Bad)
   - 200 estimators with depth 6 for complex pattern recognition
3. **Generating predictions for test data**:
   - Apply same feature engineering to test set
   - Predict binary outcome (1=Good, 0=Bad)

## Model Performance

- Validation accuracy: ~78%
- Error rate: ~22%

The model considers multiple factors:
- Customer demographics (age, location, bank)
- Current loan characteristics (amount, term, interest)
- Previous loan history (payment behavior, number of loans, amounts)

## Files

- **predict_loan_default.ipynb**: Main Jupyter notebook for training model and generating predictions
- **validate_submission.ipynb**: Jupyter notebook for validating submission format
- **requirements.txt**: Python package dependencies
- **submission.csv**: Generated predictions file (created after running prediction notebook)

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Prediction
Open and run the Jupyter notebook:
```bash
jupyter notebook predict_loan_default.ipynb
```

Or run all cells programmatically:
```bash
jupyter nbconvert --to notebook --execute predict_loan_default.ipynb
```

This will generate a `submission.csv` file with predictions in the required format:
```
customerID            Good_Bad_flag
12345667                    1
43423156                    0
```

### Validate Submission
Open and run the validation notebook:
```bash
jupyter notebook validate_submission.ipynb
```

This validates that the submission file:
- Has the correct format (customerid, Good_Bad_flag columns)
- Contains all required customer IDs
- Has valid predictions (0 or 1)
- Has no missing values

## Evaluation

The model is evaluated based on the percentage of loans predicted incorrectly (classification error rate).
