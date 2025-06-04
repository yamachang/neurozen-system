import pandas as pd
import numpy as np
import os

# Define file paths
input_file = './data/processed/data_eeg_meditation_raw_05312025.csv'
output_dir = './data/processed'
output_file = os.path.join(output_dir, 'cleaned_data.csv')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file)
print(f"Original dataset shape: {df.shape}")

# Check missing values before wrangling
print("\nChecking missing values before wrangling...")
missing_before = df.isnull().sum()
missing_percent_before = (missing_before / len(df)) * 100
missing_before_df = pd.DataFrame({
    'Missing Count': missing_before,
    'Missing Percent': missing_percent_before
})
print(missing_before_df[missing_before_df['Missing Count'] > 0])

# Check unique values in meditation_state before cleaning
print("\nUnique values in meditation_state:")
print(df['meditation_state'].value_counts())

# Remove unnecessary columns
columns_to_remove = [
    'eeg_quality_flag',
    'special_processing',
    'eeg_available',
    'ppg_available',
    'hr_available',
    'imu_available',
    'sdnn',
    'rmssd',
    'pnn50',
    'lf_power',
    'hf_power',
    'lf_hf_ratio'
]

# Check if these columns exist before removing
existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
df_cleaned = df.drop(columns=existing_columns_to_remove)
print(f"\nRemoved {len(existing_columns_to_remove)} columns: {existing_columns_to_remove}")

# Remove rows with meditation_state = -1
original_row_count = len(df_cleaned)
df_cleaned = df_cleaned[df_cleaned['meditation_state'] != -1]
removed_rows_count = original_row_count - len(df_cleaned)
print(f"\nRemoved {removed_rows_count} rows where meditation_state = -1")
print(f"Cleaned dataset shape: {df_cleaned.shape}")

# Check unique values in meditation_state after cleaning
print("\nUnique values in meditation_state after cleaning:")
print(df_cleaned['meditation_state'].value_counts())

# Check missing values after initial cleaning but before interpolation
print("\nChecking missing values after cleaning but before interpolation...")
missing_after_cleaning = df_cleaned.isnull().sum()
missing_percent_after_cleaning = (missing_after_cleaning / len(df_cleaned)) * 100
missing_after_cleaning_df = pd.DataFrame({
    'Missing Count': missing_after_cleaning,
    'Missing Percent': missing_percent_after_cleaning
})
cols_with_missing = missing_after_cleaning_df[missing_after_cleaning_df['Missing Count'] > 0]
print(f"Number of columns with missing values: {len(cols_with_missing)}")
print(f"Total missing values: {cols_with_missing['Missing Count'].sum()}")
print(f"Average missing percentage: {cols_with_missing['Missing Percent'].mean():.2f}%")
print("\nFirst few columns with missing values:")
print(cols_with_missing.head(10))

# Interpolate missing values
print("\nInterpolating missing values...")
# Group by session_id to ensure interpolation happens within each session
df_by_session = df_cleaned.groupby('session_id')

# For storing the interpolated dataframes
interpolated_dfs = []

# For each session, interpolate the missing values using linear interpolation
for session_id, session_df in df_by_session:
    # Sort by time column to ensure correct interpolation sequence
    if 'epoch_start_time_trimmed_s' in session_df.columns:
        session_df = session_df.sort_values('epoch_start_time_trimmed_s')
    
    # First, use linear interpolation method to fill gaps
    session_df_interp = session_df.interpolate(method='linear')
    
    # Then use forward fill and backward fill to handle values at the edges
    session_df_interp = session_df_interp.ffill().bfill()
    
    interpolated_dfs.append(session_df_interp)

# Combine all interpolated sessions
df_cleaned = pd.concat(interpolated_dfs)

# Check for any remaining missing values after interpolation
print("\nChecking for remaining missing values after interpolation...")
missing_after = df_cleaned.isnull().sum()
missing_percent_after = (missing_after / len(df_cleaned)) * 100
missing_after_df = pd.DataFrame({
    'Missing Count': missing_after,
    'Missing Percent': missing_percent_after
})

if missing_after.sum() > 0:
    cols_still_missing = missing_after_df[missing_after_df['Missing Count'] > 0]
    print(f"Number of columns with missing values after interpolation: {len(cols_still_missing)}")
    print(f"Total missing values after interpolation: {cols_still_missing['Missing Count'].sum()}")
    print(f"Average missing percentage after interpolation: {cols_still_missing['Missing Percent'].mean():.2f}%")
    print("\nColumns still with missing values:")
    print(cols_still_missing)
    
    # If there are still missing values, fill them with the column mean
    print("\nFilling any remaining missing values with column means...")
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                col_mean = df_cleaned[col].mean()
                df_cleaned[col] = df_cleaned[col].fillna(col_mean)
                print(f"Filled {col} with mean value: {col_mean:.4f}")
            else:
                # For non-numeric columns, use the most frequent value
                col_mode = df_cleaned[col].mode()[0]
                df_cleaned[col] = df_cleaned[col].fillna(col_mode)
                print(f"Filled {col} with mode value: {col_mode}")
    
    # Final check for missing values
    final_missing = df_cleaned.isnull().sum().sum()
    print(f"\nFinal missing values count: {final_missing}")
else:
    print("No missing values remain after interpolation")

# Save the cleaned and interpolated data
interpolated_file = os.path.join(output_dir, 'cleaned_interpolated_data.csv')
df_cleaned.to_csv(interpolated_file, index=False)
print(f"\nSaved cleaned and interpolated data to: {interpolated_file}")
print(f"Final dataset shape: {df_cleaned.shape}")