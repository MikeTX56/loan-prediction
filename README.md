# Loan Default Prediction

## Project Overview

This project focuses on predicting loan defaults using machine learning techniques. The dataset comes from the Data Science Nigeria Challenge and contains information about loan applicants, their demographics, current loan details, and historical loan performance.

## Dataset

The dataset is located in the `data-science-nigeria-challenge-1-loan-default-prediction20250307-26022-im3qg9/` directory and consists of:

- **Training Data** (4,347 customers)
  - Demographics
  - Current loan performance with target labels
  - Historical previous loan records

- **Test Data** (1,450 customers)
  - Demographics
  - Current loan performance without labels
  - Historical previous loan records

For detailed information about the dataset structure, columns, and relationships, please refer to [DATA_DESCRIPTION.md](DATA_DESCRIPTION.md).

## Challenge Description

**Goal:** Predict whether a loan will be repaid successfully (Good) or will default (Bad)

**Target Variable:** `good_bad_flag`
- Training data: "Good" or "Bad"
- Submission format: 1 (Good) or 0 (Bad)

## Data Files

| File | Records | Description |
|------|---------|-------------|
| traindemographics.csv | 4,347 | Customer demographic information |
| trainperf.csv | 4,369 | Current loan performance with target |
| trainprevloans.csv | 18,184 | Historical loan records |
| testdemographics.csv | 1,488 | Test set demographics |
| testperf.csv | 1,450 | Test set current loans |
| testprevloans.csv | 5,908 | Test set historical loans |
| SampleSubmission.csv | 1,451 | Submission template |

## Getting Started

1. Read the [DATA_DESCRIPTION.md](DATA_DESCRIPTION.md) to understand the data structure
2. Explore the training data to identify patterns
3. Engineer features from demographics, current loan, and historical loan data
4. Build and train predictive models
5. Generate predictions for the test set
6. Format predictions according to SampleSubmission.csv

## Key Features to Consider

- **Demographics**: Age, bank account type, location, employment status, education
- **Current Loan**: Amount, term, total due, interest rate (implied)
- **Loan History**: Number of previous loans, repayment patterns, loan amounts
- **Temporal**: Time between loans, trends in borrowing behavior

## License

This dataset is provided for the Data Science Nigeria Challenge.
