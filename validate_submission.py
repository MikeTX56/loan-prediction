"""
Validation script to verify submission format
"""

import pandas as pd
import sys

def validate_submission(submission_file='submission.csv'):
    """Validate the submission file format"""
    print("Validating submission file...")
    
    try:
        # Load submission
        submission = pd.read_csv(submission_file)
        
        # Check columns
        required_cols = ['customerid', 'Good_Bad_flag']
        if list(submission.columns) != required_cols:
            print(f"❌ ERROR: Expected columns {required_cols}, got {list(submission.columns)}")
            return False
        
        # Check for missing values
        if submission.isnull().any().any():
            print("❌ ERROR: Submission contains missing values")
            return False
        
        # Check Good_Bad_flag values (should be 0 or 1)
        unique_flags = submission['Good_Bad_flag'].unique()
        if not all(flag in [0, 1] for flag in unique_flags):
            print(f"❌ ERROR: Good_Bad_flag should only contain 0 or 1, found {unique_flags}")
            return False
        
        # Load test data to verify customer IDs match
        test_perf = pd.read_csv('data-science-nigeria-challenge-1-loan-default-prediction20250307-26022-im3qg9/testperf.csv')
        
        if len(submission) != len(test_perf):
            print(f"❌ ERROR: Submission has {len(submission)} rows, expected {len(test_perf)}")
            return False
        
        # Check if customer IDs match
        test_customers = set(test_perf['customerid'])
        submission_customers = set(submission['customerid'])
        
        if test_customers != submission_customers:
            missing = test_customers - submission_customers
            extra = submission_customers - test_customers
            if missing:
                print(f"❌ ERROR: Missing {len(missing)} customer IDs in submission")
            if extra:
                print(f"❌ ERROR: Found {len(extra)} extra customer IDs in submission")
            return False
        
        # All checks passed
        print("✓ Submission file format is valid")
        print(f"✓ Total predictions: {len(submission)}")
        print(f"✓ Predicted Good (1): {sum(submission['Good_Bad_flag'])}")
        print(f"✓ Predicted Bad (0): {len(submission) - sum(submission['Good_Bad_flag'])}")
        print(f"✓ Good rate: {sum(submission['Good_Bad_flag'])/len(submission)*100:.2f}%")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: File not found - {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    submission_file = 'submission.csv'
    if len(sys.argv) > 1:
        submission_file = sys.argv[1]
    
    success = validate_submission(submission_file)
    sys.exit(0 if success else 1)
