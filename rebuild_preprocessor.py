import os
import warnings
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder

warnings.filterwarnings('ignore')

SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'training datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'Model')

print("Loading data...")
df_demographics = pd.read_csv(os.path.join(DATA_PATH, 'traindemographics.csv'), parse_dates=['birthdate'])
prev_date_cols = ['approveddate', 'creationdate', 'closeddate', 'firstduedate', 'firstrepaiddate']
df_prevloans = pd.read_csv(os.path.join(DATA_PATH, 'trainprevloans.csv'), parse_dates=prev_date_cols)
perf_date_cols = ['approveddate', 'creationdate']
df_perf = pd.read_csv(os.path.join(DATA_PATH, 'trainperf.csv'), parse_dates=perf_date_cols)

target_mapping = {'Good': 0, 'Bad': 1}
if 'good_bad_flag' in df_perf.columns:
    df_perf['default_flag'] = df_perf['good_bad_flag'].map(target_mapping)
    df_perf.drop('good_bad_flag', axis=1, inplace=True)

columns_to_drop = ['longitude_gps', 'latitude_gps', 'bank_branch_clients']
existing_cols_to_drop = [col for col in columns_to_drop if col in df_demographics.columns]
df_demographics.drop(columns=existing_cols_to_drop, inplace=True)
df_demographics.drop_duplicates(subset=['customerid'], keep='last', inplace=True)
df_perf.drop_duplicates(subset=['customerid'], keep='last', inplace=True)

df_prevloans['repay_delay_days'] = (df_prevloans['firstrepaiddate'] - df_prevloans['firstduedate']).dt.days
df_prevloans['firstrepaid_late'] = (df_prevloans['repay_delay_days'] > 0).astype(int)
df_prevloans['interest_accrued'] = df_prevloans['totaldue'] - df_prevloans['loanamount']
df_prevloans['loan_per_day'] = df_prevloans['loanamount'] / df_prevloans['termdays']
df_prevloans['on_time_repayment_flag'] = (df_prevloans['repay_delay_days'] <= 0).astype(int)

prevloans_agg = df_prevloans.groupby('customerid').agg(
    total_prev_loans=('systemloanid', 'count'),
    mean_prev_loanamount=('loanamount', 'mean'),
    mean_repay_delay=('repay_delay_days', 'mean'),
    max_repay_delay=('repay_delay_days', 'max'),
    sum_late_first_repayments=('firstrepaid_late', 'sum'),
    mean_prev_interest=('interest_accrued', 'mean'),
    mean_loan_intensity=('loan_per_day', 'mean'),
    customer_total_amount_sum=('loanamount', 'sum'),
    total_due_for_prev_loans=('totaldue', 'sum'),
    sum_on_time_repayments=('on_time_repayment_flag', 'sum')
).reset_index()

prevloans_agg['late_repayment_ratio'] = prevloans_agg['sum_late_first_repayments'] / prevloans_agg['total_prev_loans']
prevloans_agg['repayment_ratio'] = prevloans_agg.apply(
    lambda row: row['customer_total_amount_sum'] / row['total_due_for_prev_loans'] if row['total_due_for_prev_loans'] != 0 else 0, axis=1)
prevloans_agg['on_time_repayment_rate'] = prevloans_agg['sum_on_time_repayments'] / prevloans_agg['total_prev_loans']
prevloans_agg.drop(columns=['total_due_for_prev_loans', 'sum_on_time_repayments'], inplace=True)

df_master = pd.merge(df_perf, prevloans_agg, on='customerid', how='left')
df_master = pd.merge(df_master, df_demographics, on='customerid', how='left')

historical_cols = [
    'total_prev_loans', 'mean_prev_loanamount', 'mean_repay_delay',
    'max_repay_delay', 'sum_late_first_repayments', 'mean_prev_interest',
    'late_repayment_ratio', 'mean_loan_intensity',
    'customer_total_amount_sum', 'repayment_ratio', 'on_time_repayment_rate'
]
df_master[historical_cols] = df_master[historical_cols].fillna(0)
df_master['age_at_loan'] = (df_master['creationdate'] - df_master['birthdate']).dt.days / 365.25
df_master['age_at_loan'] = df_master['age_at_loan'].fillna(df_master['age_at_loan'].median())

cols_to_drop = ['customerid', 'systemloanid', 'approveddate', 'creationdate', 'referredby', 'birthdate']
df_master.drop(columns=[col for col in cols_to_drop if col in df_master.columns], inplace=True)

X = df_master.drop(columns='default_flag')
y = df_master['default_flag']

X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

numeric_features = X_train_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
all_categorical_features = X_train_raw.select_dtypes(include=['object']).columns.tolist()

target_encode_features = ['bank_name_clients']
onehot_encode_features = [f for f in all_categorical_features if f not in target_encode_features]

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
onehot_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
target_encoder_transformer = Pipeline(steps=[('target_encoder', TargetEncoder(handle_unknown='value', handle_missing='value'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('target_cat', target_encoder_transformer, target_encode_features),
        ('onehot_cat', onehot_transformer, onehot_encode_features)
    ],
    remainder='passthrough'
)

print("Fitting preprocessor...")
preprocessor.fit(X_train_raw, y_train)

# Fix feature_names order to match the ones we expect in webapp
expected_order = numeric_features + target_encode_features + onehot_encode_features

# Save it to disk
out_path = os.path.join(MODEL_DIR, 'loan_preprocessor.joblib')
joblib.dump(preprocessor, out_path)
print(f"Successfully rebuilt and saved preprocessor to {out_path}")
