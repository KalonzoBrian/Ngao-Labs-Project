import streamlit as st
from model_handler import predict_default

# --- Page Config ---
st.set_page_config(
    page_title="AgriLoan Predictor",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Dark Mode State ---
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# --- Dark Mode Toggle at Top ---
toggle_col1, toggle_col2 = st.columns([6, 1])
with toggle_col1:
    st.title("🌾 Agricultural Micro-Loan Predictor")
with toggle_col2:
    dark_mode = st.toggle("🌙", value=st.session_state.dark_mode, key="dark_toggle")
    st.session_state.dark_mode = dark_mode

st.markdown("Empowering Kenyan Farmers and SACCOs with AI-driven credit decisions.")

# --- Theme Colors ---
if st.session_state.dark_mode:
    bg_color = "#1a1a2e"
    card_bg = "#16213e"
    card_border = "#2a2a4a"
    text_color = "#e0e0e0"
    text_secondary = "#b0b0b0"
    text_muted = "#8888aa"
    heading_color = "#66bb6a"
    bar_track = "#2a2a4a"
    summary_bg = "#1b2e1b"
    summary_border = "#4CAF50"
    prediction_good_bg = "#1b3d1b"
    prediction_good_text = "#66bb6a"
    prediction_bad_bg = "#3d1b1b"
    prediction_bad_text = "#ef5350"
    hover_shadow = "rgba(0,0,0,0.4)"
    sidebar_bg = "#16213e"
    input_bg = "#1a1a2e"
else:
    bg_color = "#f9fbf9"
    card_bg = "#ffffff"
    card_border = "#e0e0e0"
    text_color = "#1b1b1b"
    text_secondary = "#757575"
    text_muted = "#9e9e9e"
    heading_color = "#2E7D32"
    bar_track = "#f0f0f0"
    summary_bg = "#f1f8e9"
    summary_border = "#2E7D32"
    prediction_good_bg = "#e8f5e9"
    prediction_good_text = "#2E7D32"
    prediction_bad_bg = "#ffebee"
    prediction_bad_text = "#c62828"
    hover_shadow = "rgba(0,0,0,0.1)"
    sidebar_bg = "#e8f5e9"
    input_bg = "#ffffff"

# --- Custom CSS for Styling ---
st.markdown(f"""
    <style>
    /* --- Global Theme --- */
    .main {{
        background-color: {bg_color};
    }}
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
    }}
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
    }}
    [data-testid="stHeader"] {{
        background-color: {bg_color};
    }}
    .stMarkdown, .stMarkdown p, [data-testid="stMarkdownContainer"] p {{
        color: {text_color};
    }}
    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: #45a049;
    }}
    h1, h2, h3 {{
        color: {heading_color} !important;
    }}
    /* --- Input Elements Dark Mode --- */
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stExpander"] summary span {{
        color: {text_color} !important;
    }}
    .prediction-good {{
        color: {prediction_good_text};
        font-weight: bold;
        font-size: 24px;
        padding: 10px;
        background-color: {prediction_good_bg};
        border-radius: 5px;
        text-align: center;
    }}
    .prediction-bad {{
        color: {prediction_bad_text};
        font-weight: bold;
        font-size: 24px;
        padding: 10px;
        background-color: {prediction_bad_bg};
        border-radius: 5px;
        text-align: center;
    }}
    /* --- SHAP Explanation Styles --- */
    .shap-section-title {{
        font-size: 20px;
        font-weight: bold;
        color: {heading_color};
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }}
    .shap-card {{
        display: flex;
        align-items: center;
        padding: 10px 14px;
        margin-bottom: 6px;
        border-radius: 8px;
        background-color: {card_bg};
        border: 1px solid {card_border};
        transition: box-shadow 0.2s;
    }}
    .shap-card:hover {{
        box-shadow: 0 2px 8px {hover_shadow};
    }}
    .shap-rank {{
        font-size: 14px;
        font-weight: bold;
        color: {text_muted};
        min-width: 28px;
        text-align: center;
    }}
    .shap-info {{
        flex: 1;
        padding: 0 12px;
    }}
    .shap-feature-name {{
        font-weight: 600;
        font-size: 14px;
        color: {text_color};
    }}
    .shap-feature-value {{
        font-size: 12px;
        color: {text_secondary};
        margin-top: 2px;
    }}
    .shap-bar-container {{
        width: 150px;
        min-width: 150px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .shap-bar-track {{
        flex: 1;
        height: 10px;
        background-color: {bar_track};
        border-radius: 5px;
        overflow: hidden;
    }}
    .shap-bar-fill-risk {{
        height: 100%;
        background: linear-gradient(90deg, #ef5350, #c62828);
        border-radius: 5px;
        transition: width 0.4s ease;
    }}
    .shap-bar-fill-safe {{
        height: 100%;
        background: linear-gradient(90deg, #66bb6a, #2E7D32);
        border-radius: 5px;
        transition: width 0.4s ease;
    }}
    .shap-impact-label {{
        font-size: 11px;
        font-weight: bold;
        min-width: 70px;
        text-align: right;
    }}
    .shap-risk-label {{ color: #ef5350; }}
    .shap-safe-label {{ color: #66bb6a; }}
    .shap-legend {{
        display: flex;
        gap: 20px;
        margin: 8px 0 16px 0;
        font-size: 13px;
        color: {text_secondary};
    }}
    .shap-legend-dot {{
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 5px;
        vertical-align: middle;
    }}
    .shap-summary-text {{
        font-size: 14px;
        color: {text_color};
        line-height: 1.7;
        padding: 12px 16px;
        background-color: {summary_bg};
        border-left: 4px solid {summary_border};
        border-radius: 4px;
        margin-top: 10px;
    }}
    /* --- Dark Mode Toggle Styling --- */
    [data-testid="stToggle"] label span {{
        font-size: 22px !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.divider()

# --- Input Sections ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Loan Request Details")
    loanamount = st.number_input("Requested Loan Amount (KES)", min_value=1000.0, value=10000.0, step=500.0)
    termdays = st.selectbox("Term Days", [15, 30, 60, 90])
    # Total due is generally Loan Amount + Interest. We'll approximate a 15% interest if not specified.
    totaldue = st.number_input("Expected Total Due (KES)", min_value=1000.0, value=loanamount * 1.15, step=500.0)

with col2:
    st.subheader("Applicant Demographics")
    age_at_loan = st.number_input("Age of Applicant", min_value=18, max_value=100, value=30)
    employment = st.selectbox("Employment Status", ["Permanent", "Self-Employed", "Student", "Unemployed", "Retired"])
    education = st.selectbox("Level of Education", ["Primary", "Secondary", "Graduate", "Post-Graduate", "None"])
    
st.subheader("Banking Details")
b_col1, b_col2 = st.columns(2)
with b_col1:
    bank_name = st.selectbox("Bank Name", ["GT Bank", "Equity Bank", "KCB", "Co-operative Bank", "Family Bank", "Access Bank", "UBA", "Other"])
with b_col2:
    account_type = st.selectbox("Account Type", ["Savings", "Current", "Other"])

st.divider()

with st.expander("📊 Advanced Credit History (Optional)"):
    st.markdown("For returning customers, provide historical data if available. Otherwise, leave as default.")
    total_prev_loans = st.number_input("Total Previous Loans", min_value=0, value=0)
    mean_prev_loanamount = st.number_input("Average Previous Loan Amount", min_value=0.0, value=0.0)
    mean_repay_delay = st.number_input("Average Repayment Delay (Days)", value=0.0)
    max_repay_delay = st.number_input("Max Repayment Delay (Days)", value=0.0)
    sum_late_first_repayments = st.number_input("Count of Late First Repayments", min_value=0, value=0)
    mean_prev_interest = st.number_input("Average Previous Interest Accrued", min_value=0.0, value=0.0)
    
    # These are calculated features, but we can set them to 0 as default if no previous loans
    late_repayment_ratio = sum_late_first_repayments / total_prev_loans if total_prev_loans > 0 else 0.0
    mean_loan_intensity = mean_prev_loanamount / 30.0 if mean_prev_loanamount > 0 else 0.0
    customer_total_amount_sum = total_prev_loans * mean_prev_loanamount
    repayment_ratio = 1.0 if total_prev_loans > 0 else 0.0
    on_time_repayment_rate = 1.0 if total_prev_loans > 0 and sum_late_first_repayments == 0 else 0.0

st.divider()

# --- Prediction Action ---
if st.button("Predict Default Probability", use_container_width=True):
    # Assemble input dictionary
    input_data = {
        'loannumber': float(total_prev_loans) + 1.0,
        'loanamount': loanamount,
        'totaldue': totaldue,
        'termdays': termdays,
        'bank_account_type': account_type,
        'bank_name_clients': bank_name,
        'employment_status_clients': employment,
        'level_of_education_clients': education,
        'age_at_loan': float(age_at_loan),
        
        # History
        'total_prev_loans': float(total_prev_loans),
        'mean_prev_loanamount': float(mean_prev_loanamount),
        'mean_repay_delay': float(mean_repay_delay),
        'max_repay_delay': float(max_repay_delay),
        'sum_late_first_repayments': float(sum_late_first_repayments),
        'mean_prev_interest': float(mean_prev_interest),
        'late_repayment_ratio': float(late_repayment_ratio),
        'mean_loan_intensity': float(mean_loan_intensity),
        'customer_total_amount_sum': float(customer_total_amount_sum),
        'repayment_ratio': float(repayment_ratio),
        'on_time_repayment_rate': float(on_time_repayment_rate)
    }
    
    with st.spinner("Analyzing applicant data..."):
        try:
            prob, shap_explanation = predict_default(input_data)
            
            # Display results
            if prob > 0.5:
                st.markdown(f'<div class="prediction-bad">High Risk of Default! ({prob*100:.1f}%)</div>', unsafe_allow_html=True)
                st.warning("The model predicts this applicant is highly likely to default. Proceed with caution or request additional collateral.")
            else:
                st.markdown(f'<div class="prediction-good">Low Risk / Good Loan ({100 - prob*100:.1f}% Confidence)</div>', unsafe_allow_html=True)
                st.success("The model predicts this applicant is likely to repay on time.")
            
            # --- SHAP Explanation Section ---
            if shap_explanation:
                st.divider()
                st.markdown('<div class="shap-section-title">🔍 AI Explanation — Why This Decision?</div>', unsafe_allow_html=True)
                st.markdown("""
                    <div class="shap-legend">
                        <span><span class="shap-legend-dot" style="background:#c62828;"></span> Increases default risk</span>
                        <span><span class="shap-legend-dot" style="background:#2E7D32;"></span> Decreases default risk</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show top factors (limit to 10 most impactful to keep it readable)
                top_factors = shap_explanation[:10]
                
                # Compute max absolute SHAP value for scaling bars
                max_abs_shap = max(abs(f['shap_value']) for f in top_factors) if top_factors else 1.0
                
                for i, factor in enumerate(top_factors, 1):
                    shap_val = factor['shap_value']
                    abs_shap = abs(shap_val)
                    bar_pct = min((abs_shap / max_abs_shap) * 100, 100) if max_abs_shap > 0 else 0
                    
                    # Positive SHAP = pushes toward default (risk)
                    is_risk = shap_val > 0
                    bar_class = "shap-bar-fill-risk" if is_risk else "shap-bar-fill-safe"
                    label_class = "shap-risk-label" if is_risk else "shap-safe-label"
                    impact_text = "↑ Risk" if is_risk else "↓ Safe"
                    
                    # Format the raw value nicely
                    raw_val = factor['value']
                    if isinstance(raw_val, float):
                        display_val = f"{raw_val:,.2f}"
                    else:
                        display_val = str(raw_val)
                    
                    st.markdown(f"""
                        <div class="shap-card">
                            <div class="shap-rank">#{i}</div>
                            <div class="shap-info">
                                <div class="shap-feature-name">{factor['label']}</div>
                                <div class="shap-feature-value">Value: {display_val}</div>
                            </div>
                            <div class="shap-bar-container">
                                <div class="shap-bar-track">
                                    <div class="{bar_class}" style="width: {bar_pct:.0f}%"></div>
                                </div>
                                <div class="shap-impact-label {label_class}">{impact_text}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # --- Narrative Summary ---
                risk_factors = [f for f in shap_explanation[:5] if f['shap_value'] > 0]
                safe_factors = [f for f in shap_explanation[:5] if f['shap_value'] <= 0]
                
                summary_parts = []
                if risk_factors:
                    risk_names = ", ".join([f"**{f['label']}**" for f in risk_factors[:3]])
                    summary_parts.append(f"⚠️ The main factors **increasing** default risk are: {risk_names}.")
                if safe_factors:
                    safe_names = ", ".join([f"**{f['label']}**" for f in safe_factors[:3]])
                    summary_parts.append(f"✅ The factors **reducing** risk are: {safe_names}.")
                
                if summary_parts:
                    st.markdown("---")
                    st.markdown("#### 📋 Summary")
                    for part in summary_parts:
                        st.markdown(part)
                
                # --- Responsible AI Disclaimer ---
                st.markdown("---")
                st.info(
                    "🤖 **Responsible AI Notice**: This explanation uses SHAP (SHapley Additive exPlanations) "
                    "to show how each factor contributed to the model's prediction. The analysis is provided "
                    "for transparency and should be used as a **decision-support tool**, not as a sole basis "
                    "for credit decisions. Always apply human judgment, institutional policies, and regulatory "
                    "guidelines when making lending decisions."
                )
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

