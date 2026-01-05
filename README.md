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
1. Data preprocessing and feature engineering
2. Training a classification model (XGBoost)
3. Generating predictions for test data

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Prediction
```bash
python predict_loan_default.py
```

This will generate a `submission.csv` file with predictions in the required format:
```
customerID            Good_Bad_flag
12345667                    1
43423156                    0
```

## Evaluation

The model is evaluated based on the percentage of loans predicted incorrectly (classification error rate).
