import pandas as pd

def process_data(df):
    """
    This function processes the input DataFrame by converting datetime columns, extracting relevant features,
    and dropping unnecessary and redundant columns.
    """

    # Keep only rows where mean_airtime_balance_20_weeks is positive
    df = df.loc[df["mean_airtime_balance_20_weeks"] >= 0]

    # Convert the 'request_datetime' and 'reimbursement_date' columns to datetime objects:
    df['request_datetime'] = pd.to_datetime(df['request_datetime'])
    df['reimbursement_date'] = pd.to_datetime(df['reimbursement_date'])

    # Extract relevant features from the datetime columns 
    df['request_year'] = df['request_datetime'].dt.year
    df['request_month'] = df['request_datetime'].dt.month
    df['request_day'] = df['request_datetime'].dt.day
    df['request_hour'] = df['request_datetime'].dt.hour
    df['request_dayofweek'] = df['request_datetime'].dt.dayofweek
    df['reimbursement_year'] = df['reimbursement_date'].dt.year
    df['reimbursement_month'] = df['reimbursement_date'].dt.month
    df['reimbursement_day'] = df['reimbursement_date'].dt.day
    df['reimbursement_dayofweek'] = df['reimbursement_date'].dt.dayofweek
    df['date_diff'] = (df['reimbursement_date'] - df['request_datetime']).dt.days

    # Extract relevant features from other columns
    epsilon = 1e-8 # Define a small constant to avoid division by zero
    df['mean_cashout_to_balance_ratio'] = df['mean_volcashout_20_weeks'] / (df['mean_balance_20_weeks'] + epsilon)
    df['mean_cashout_to_airtime_ratio'] = df['mean_volcashout_20_weeks'] / (df['mean_airtime_balance_20_weeks'] + epsilon)
    df['mean_volotherout_to_balance_ratio'] = df['mean_volotherout'] / (df['mean_balance_20_weeks'] + epsilon)
    df['algo_diff'] = abs(df['algo1_eligible_amount'] - df['algo2_eligible_amount'])
    df['total_eligible_amount'] = df['algo1_eligible_amount'] + df['algo2_eligible_amount']
    df['min_algo_eligible_amount'] = df[['algo1_eligible_amount', 'algo2_eligible_amount']].min(axis=1)
    df['max_algo_eligible_amount'] = df[['algo1_eligible_amount', 'algo2_eligible_amount']].max(axis=1)
    df['mean_algo_eligible_amount'] = df[['algo1_eligible_amount', 'algo2_eligible_amount']].mean(axis=1)

    # Drop unnecessary and redundant columns
    df.drop(['request_datetime', 'reimbursement_date'], axis=1, inplace=True) # Drop the original datetime columns
    id_columns = ['customer_id', 'simulation_id', 'loan_id'] # drop ID columns
    df.drop(id_columns, axis=1, inplace=True)
    return df
