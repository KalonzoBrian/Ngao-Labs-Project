import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import category_encoders # Required for the target encoder in the pipeline

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'Model')

PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'loan_preprocessor.joblib')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_tuned_baseline.json')

# We load models at module level to avoid reloading on every prediction
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    # Get the underlying Booster for native SHAP computation
    _booster = xgb_model.get_booster()
except Exception as e:
    print(f"Error loading models: {e}")
    preprocessor = None
    xgb_model = None
    _booster = None

# Expected features in the exact order/format
EXPECTED_FEATURES = [
    'loannumber', 'loanamount', 'totaldue', 'termdays', 
    'bank_account_type', 'bank_name_clients', 'employment_status_clients', 'level_of_education_clients',
    'total_prev_loans', 'mean_prev_loanamount', 'mean_repay_delay', 'max_repay_delay', 
    'sum_late_first_repayments', 'mean_prev_interest', 'late_repayment_ratio', 'mean_loan_intensity',
    'customer_total_amount_sum', 'repayment_ratio', 'on_time_repayment_rate', 'age_at_loan'
]

# Human-readable labels for each feature (used in SHAP explanations)
FEATURE_LABELS = {
    'loannumber': 'Loan Number',
    'loanamount': 'Loan Amount',
    'totaldue': 'Total Due',
    'termdays': 'Term Days',
    'bank_account_type': 'Bank Account Type',
    'bank_name_clients': 'Bank Name',
    'employment_status_clients': 'Employment Status',
    'level_of_education_clients': 'Level of Education',
    'total_prev_loans': 'Total Previous Loans',
    'mean_prev_loanamount': 'Avg. Previous Loan Amount',
    'mean_repay_delay': 'Avg. Repayment Delay',
    'max_repay_delay': 'Max Repayment Delay',
    'sum_late_first_repayments': 'Late First Repayments Count',
    'mean_prev_interest': 'Avg. Previous Interest',
    'late_repayment_ratio': 'Late Repayment Ratio',
    'mean_loan_intensity': 'Loan Intensity',
    'customer_total_amount_sum': 'Customer Total Loan History',
    'repayment_ratio': 'Repayment Ratio',
    'on_time_repayment_rate': 'On-Time Repayment Rate',
    'age_at_loan': 'Age at Loan Application'
}


def predict_default(input_data: dict):
    """
    Takes a dictionary of input features, preprocesses it, and returns:
      - prob: the probability of default (float)
      - shap_explanation: a list of dicts with keys 'feature', 'label', 'value', 'shap_value'
        sorted by absolute SHAP impact (descending). Positive SHAP = pushes toward default.
    """
    if preprocessor is None or xgb_model is None:
        raise ValueError("Models are not loaded correctly.")
        
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Ensure all expected columns are present
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            if col in ['bank_account_type', 'bank_name_clients', 'employment_status_clients', 'level_of_education_clients']:
                df[col] = 'Unknown'
            else:
                df[col] = 0.0
                
    # Order columns
    df = df[EXPECTED_FEATURES]
    
    # Preprocess
    X_processed = preprocessor.transform(df)
    
    # Predict Probability (Class 1 = Default/Bad)
    prob = xgb_model.predict_proba(X_processed)[0][1]
    
    # --- SHAP Explanation (using XGBoost's native pred_contribs) ---
    shap_explanation = []
    if _booster is not None:
        try:
            # Convert to DMatrix for native SHAP
            if isinstance(X_processed, pd.DataFrame):
                dmatrix = xgb.DMatrix(X_processed.values)
            else:
                dmatrix = xgb.DMatrix(X_processed)
            
            # pred_contribs returns shape (n_samples, n_features + 1)
            # Last column is the bias (base value). Others are per-feature SHAP values.
            contribs = _booster.predict(dmatrix, pred_contribs=True)
            shap_values = contribs[0]  # First (and only) sample
            
            # The last value is the bias term — exclude it from feature explanations
            feature_shap_values = shap_values[:-1]
            
            # Get processed feature names
            if hasattr(X_processed, 'columns'):
                processed_feature_names = list(X_processed.columns)
            else:
                processed_feature_names = _get_processed_feature_names()
            
            # Aggregate SHAP values back to the original feature level
            aggregated_shap = _aggregate_shap_to_original(feature_shap_values, processed_feature_names, df)
            
            for feature_name, shap_val, raw_val in aggregated_shap:
                shap_explanation.append({
                    'feature': feature_name,
                    'label': FEATURE_LABELS.get(feature_name, feature_name),
                    'value': raw_val,
                    'shap_value': float(shap_val)
                })
            
            # Sort by absolute SHAP impact (most influential first)
            shap_explanation.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        except Exception as e:
            print(f"SHAP explanation error: {e}")
            # Return prediction without SHAP if it fails
    
    return float(prob), shap_explanation


def _get_processed_feature_names():
    """Attempt to extract feature names from the preprocessor pipeline."""
    try:
        if hasattr(preprocessor, 'get_feature_names_out'):
            return list(preprocessor.get_feature_names_out())
        # For ColumnTransformer
        if hasattr(preprocessor, 'transformers_'):
            names = []
            for name, trans, cols in preprocessor.transformers_:
                if name == 'remainder' and trans == 'drop':
                    continue
                if hasattr(trans, 'get_feature_names_out'):
                    names.extend(trans.get_feature_names_out(cols if isinstance(cols, list) else [cols]))
                elif isinstance(cols, list):
                    names.extend(cols)
                else:
                    names.append(cols)
            return names
    except Exception:
        pass
    # Fallback: use EXPECTED_FEATURES (works when preprocessor preserves feature count)
    return EXPECTED_FEATURES


def _aggregate_shap_to_original(shap_values, processed_names, raw_df):
    """
    Aggregate SHAP values from processed feature space back to original features.
    Returns list of (original_feature_name, aggregated_shap_value, raw_value).
    """
    # If the number of SHAP values matches expected features, it's a 1-to-1 mapping
    if len(shap_values) == len(EXPECTED_FEATURES):
        result = []
        for i, feat in enumerate(EXPECTED_FEATURES):
            raw_val = raw_df[feat].iloc[0] if feat in raw_df.columns else 'N/A'
            result.append((feat, shap_values[i], raw_val))
        return result
    
    # Map processed names back to original features
    # ColumnTransformer adds prefixes like 'num__', 'target_cat__', 'onehot_cat__'
    aggregated = {}
    for i, p_name in enumerate(processed_names):
        if i >= len(shap_values):
            break
        
        # Strip ColumnTransformer prefix (e.g. 'num__age_at_loan' -> 'age_at_loan')
        stripped_name = p_name
        if '__' in p_name:
            stripped_name = p_name.split('__', 1)[1]
        
        matched = False
        for orig_feat in EXPECTED_FEATURES:
            # Match exact name, or prefix for one-hot encoded features (e.g. 'bank_account_type_Savings')
            if stripped_name == orig_feat or stripped_name.startswith(orig_feat + '_'):
                if orig_feat not in aggregated:
                    aggregated[orig_feat] = 0.0
                aggregated[orig_feat] += shap_values[i]
                matched = True
                break
        if not matched:
            # Keep as-is if we can't map it
            aggregated[stripped_name] = aggregated.get(stripped_name, 0.0) + shap_values[i]
    
    result = []
    for feat, sv in aggregated.items():
        raw_val = raw_df[feat].iloc[0] if feat in raw_df.columns else 'N/A'
        result.append((feat, sv, raw_val))
    
    return result

