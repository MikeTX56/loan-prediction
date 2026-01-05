# Loan Default Prediction - Data Description

## Overview

This dataset is from the Data Science Nigeria Challenge for predicting loan defaults. The goal is to predict whether a customer will default on their loan (bad) or successfully repay it (good) based on their demographic information, current loan performance, and previous loan history.

## Dataset Files

### 1. Training Data

#### traindemographics.csv
Contains demographic information about customers in the training set.

**Columns:**
- `customerid`: Unique identifier for each customer
- `birthdate`: Customer's date of birth (format: YYYY-MM-DD HH:MM:SS.ffffff)
- `bank_account_type`: Type of bank account (e.g., Savings, Current)
- `longitude_gps`: GPS longitude coordinate of customer location
- `latitude_gps`: GPS latitude coordinate of customer location
- `bank_name_clients`: Name of the customer's bank (e.g., GT Bank, Sterling Bank, Fidelity Bank)
- `bank_branch_clients`: Bank branch (often empty)
- `employment_status_clients`: Employment status (e.g., Permanent, Temporary, or empty)
- `level_of_education_clients`: Customer's education level (often empty)

**Size:** 4,347 records

#### trainperf.csv
Contains current loan performance data for the training set. **This file includes the target variable.**

**Columns:**
- `customerid`: Unique identifier linking to demographics
- `systemloanid`: System-generated loan identifier
- `loannumber`: Sequential loan number for the customer
- `approveddate`: Date and time when loan was approved
- `creationdate`: Date and time when loan was created
- `loanamount`: Principal loan amount
- `totaldue`: Total amount due (principal + interest)
- `termdays`: Loan term in days (e.g., 15, 30)
- `referredby`: Referral information (often empty)
- `good_bad_flag`: **Target variable** - "Good" (loan repaid) or "Bad" (loan defaulted)

**Size:** 4,369 records

#### trainprevloans.csv
Contains historical data about previous loans for customers in the training set.

**Columns:**
- `customerid`: Unique identifier linking to demographics
- `systemloanid`: System-generated loan identifier
- `loannumber`: Sequential loan number for the customer
- `approveddate`: Date and time when loan was approved
- `creationdate`: Date and time when loan was created
- `loanamount`: Principal loan amount
- `totaldue`: Total amount due (principal + interest)
- `termdays`: Loan term in days
- `closeddate`: Date and time when loan was closed/completed
- `referredby`: Referral information (often empty)
- `firstduedate`: Date when first payment was due
- `firstrepaiddate`: Date when first payment was made

**Size:** 18,184 records (multiple previous loans per customer)

### 2. Test Data

#### testdemographics.csv
Demographic information for test set customers (same structure as traindemographics.csv).

**Size:** 1,488 records

#### testperf.csv
Current loan performance data for test set (same structure as trainperf.csv but **without the target variable** `good_bad_flag`).

**Size:** 1,450 records

#### testprevloans.csv
Historical loan data for test set customers (same structure as trainprevloans.csv).

**Size:** 5,908 records

### 3. Submission File

#### SampleSubmission.csv
Template for submitting predictions.

**Columns:**
- `customerid`: Customer identifier from test set
- `Good_Bad_flag`: Predicted target (1 for Good, 0 for Bad)

**Size:** 1,451 records

**Note:** The target variable in SampleSubmission uses numeric encoding (1/0) while trainperf.csv uses text labels ("Good"/"Bad").

## Key Relationships

1. **One-to-One**: Each `customerid` has one record in demographics and one current loan in perf files
2. **One-to-Many**: Each `customerid` may have multiple previous loans in prevloans files
3. **Link by**: `customerid` is the primary key connecting all tables

## Data Characteristics

### Date Formats
All dates are in format: `YYYY-MM-DD HH:MM:SS.ffffff`

### Numeric Formats
Loan amounts use decimal format with 4 decimal places (e.g., 30000.0000)

### Missing Values
Several fields contain empty values, particularly:
- `bank_branch_clients`
- `employment_status_clients`
- `level_of_education_clients`
- `referredby`

### Geographic Coverage
GPS coordinates suggest data is from Nigeria (latitude ~5-7°N, longitude ~3-5°E)

## Prediction Task

**Objective:** Predict the `good_bad_flag` for loans in the test set

**Target Variable:**
- In training: "Good" or "Bad" (text)
- In submission: 1 (Good) or 0 (Bad) (numeric)

**Evaluation:** Success is measured by prediction accuracy on the test set

## Data Usage Tips

1. **Feature Engineering**: Consider aggregating previous loan statistics (count, average amount, repayment patterns)
2. **Time Features**: Extract age from birthdate, loan duration, time between loans
3. **Financial Ratios**: Calculate debt-to-income proxies from loan amounts
4. **Geographic Features**: Use GPS coordinates for regional patterns
5. **Loan History**: Previous loan performance may be a strong predictor
6. **Handle Missing Data**: Many demographic fields have missing values requiring imputation or special handling
