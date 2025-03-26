import pandas as pd
import ydata_profiling as dp
from pathlib import Path
import os

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)

# Define file paths
data_dir = Path(r"C:\Users\Shekar\Desktop\Shekar\GCP")
mock_data_path = data_dir / "mock_data.csv"
currency_mapping_path = data_dir / "country-code-to-currency-code-mapping.csv"

# Verify files exist before loading
if not mock_data_path.exists():
    raise FileNotFoundError(f"Mock data file not found at {mock_data_path}")
if not currency_mapping_path.exists():
    raise FileNotFoundError(f"Currency mapping file not found at {currency_mapping_path}")

# Load datasets with error handling
try:
    sample_df = pd.read_csv(mock_data_path)
    cur_cd_df = pd.read_csv(currency_mapping_path)
except Exception as e:
    raise Exception(f"Error loading CSV files: {str(e)}")

# Data Cleaning and Preparation
def clean_data(df):
    """Perform basic data cleaning"""
    # Convert column names to lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip()
    
    # Convert string columns to uppercase and strip whitespace
    string_cols = df.select_dtypes(include='object').columns
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()
    
    return df

sample_df = clean_data(sample_df)
cur_cd_df = clean_data(cur_cd_df)

# Standardize column names in currency mapping
cur_cd_df = cur_cd_df.rename(columns={
    'currency': 'curr', 
    'code': 'expected_currency_code',
    'countrycode': 'country_code'
})

# Handle duplicates in currency mapping data
print("\nChecking for duplicates in currency mapping data...")
duplicates = cur_cd_df.duplicated(subset=['country_code'], keep=False)
if duplicates.any():
    print(f"Found {duplicates.sum()} duplicate country codes in currency mapping data")
    print("Keeping only the first occurrence of each country code")
    cur_cd_df = cur_cd_df.drop_duplicates(subset=['country_code'], keep='first')

# Merge datasets with validation
if 'country' not in sample_df.columns:
    raise ValueError("'country' column not found in mock data")
if 'country_code' not in cur_cd_df.columns:
    raise ValueError("'country_code' column not found in currency mapping data")

enriched_df = pd.merge(
    sample_df, 
    cur_cd_df, 
    left_on='country', 
    right_on='country_code', 
    how='left'
)

# Data Profiling Rules (same as before)
def rule1(row):
    """Check if transaction amount differs from reported amount with currency mismatch"""
    if pd.isna(row['transaction_amount']) or pd.isna(row['reported_amount']):
        return False
    
    if row['transaction_amount'] != row['reported_amount']:
        if row['currency'] != row['expected_currency_code']:
            if abs(row['transaction_amount'] - row['reported_amount']) > (row['transaction_amount'] * 0.01):
                return True
    return False

def rule2(row):
    """Check if transaction currency matches country's expected currency"""
    if pd.isna(row['currency']) or pd.isna(row['expected_currency_code']):
        return False
    return row['currency'] != row['expected_currency_code']

def rule3(row):
    """Identify unusually large transactions (> 1M)"""
    if pd.isna(row['transaction_amount']):
        return False
    return row['transaction_amount'] > 1000000

def rule4(row):
    """Check for negative account balances"""
    if pd.isna(row['account_balance']):
        return False
    return row['account_balance'] < 0

def rule5(row):
    """Identify high risk transactions with large amounts"""
    if pd.isna(row['risk_score']) or pd.isna(row['transaction_amount']):
        return False
    return (row['risk_score'] > 75) and (row['transaction_amount'] > 50000)

def rule6(row):
    """Check for transactions in unusual currencies for the account type"""
    if pd.isna(row['account_type']) or pd.isna(row['currency']):
        return False
    
    unusual_currencies = {
        'SAVINGS': ['BTC', 'XRP'],
        'CHECKING': ['XAU', 'XAG'],
        'BUSINESS': []
    }
    
    account_type = row['account_type']
    if account_type in unusual_currencies:
        return row['currency'] in unusual_currencies[account_type]
    return False

# Apply rules
rules = {
    'rule1_amount_mismatch': rule1,
    'rule2_currency_mismatch': rule2,
    'rule3_large_transaction': rule3,
    'rule4_negative_balance': rule4,
    'rule5_high_risk_large_amount': rule5,
    'rule6_unusual_currency': rule6
}

for col, func in rules.items():
    enriched_df[col] = enriched_df.apply(func, axis=1)



# Modified generate_report function
def generate_report(df):
    """Generate data profiling report"""
    print("\nData Profile Summary:")
    print(f"Total records: {len(df):,}")
    
    summary_data = []
    for rule in rules.keys():
        count = df[rule].sum()
        summary_data.append({
            'Rule': rule,
            'Count': count,
            'Percentage': f"{(count/len(df))*100:.2f}%"
        })
    
    print(pd.DataFrame(summary_data).to_string(index=False))
    
    # Generate ydata-profiling report
    print("\nGenerating detailed profiling report...")
    profile = dp.ProfileReport(
        df, 
        title="Transaction Data Profiling Report",
        explorative=True
    )
    
    report_path = data_dir / "transaction_data_profile_report.html"
    profile.to_file(report_path)
    print(f"\nDetailed report saved to: {report_path}")

    # Show sample flagged records with column existence checks
    flagged_cols = list(rules.keys())
    flagged_df = df[df[flagged_cols].any(axis=1)]
    
    # Define columns to display and check their existence
    display_cols = []
    for col in ['customer_id', 'country', 'currency', 'expected_currency_code',
               'account_balance', 'transaction_amount', 'risk_score', 'account_type']:
        if col in flagged_df.columns:
            display_cols.append(col)
    
    display_cols.extend(flagged_cols)
    
    if not flagged_df.empty:
        print("\nSample of flagged records (5 random samples):")
        print(flagged_df.sample(min(5, len(flagged_df)))[display_cols].to_string(index=False))
    else:
        print("\nNo records were flagged by any rule.")

# Additional debugging check before generating report
print("\nAvailable columns in enriched_df:", list(enriched_df.columns))
generate_report(enriched_df)