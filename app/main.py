import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Credit Risk Assessment | Lauki Finance",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Card styling */
    .css-1r6slb0, .css-1r6slb0 {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e8ecf1;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1a2332 0%, #2c3e50 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1rem;
        opacity: 0.85;
        font-weight: 300;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #7f8c8d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-top: 0.25rem;
    }
    
    .metric-value.good {
        color: #27ae60;
    }
    
    .metric-value.warning {
        color: #f39c12;
    }
    
    .metric-value.danger {
        color: #e74c3c;
    }
    
    /* Input section styling */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ecf0f1;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 8px;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
    }
    
    /* Results section */
    .result-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-top: 2rem;
        border: 1px solid #e8ecf1;
    }
    
    /* Rating badges */
    .rating-badge {
        display: inline-block;
        padding: 0.5rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        color: white;
        text-align: center;
    }
    
    .rating-excellent {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
    }
    
    .rating-good {
        background: linear-gradient(135deg, #3498db, #5dade2);
    }
    
    .rating-average {
        background: linear-gradient(135deg, #f39c12, #f1c40f);
    }
    
    .rating-poor {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #ecf0f1;
        text-align: center;
        color: #95a5a6;
        font-size: 0.85rem;
    }
    
    /* Tooltip hover effect */
    .info-tooltip {
        color: #7f8c8d;
        font-size: 0.8rem;
        cursor: help;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .header-title {
            font-size: 1.8rem;
        }
        .metric-value {
            font-size: 1.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load model and components
MODEL_PATH = 'artifacts/model_data.joblib'
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']

def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                    delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                    loan_purpose, loan_type):
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }
    df = pd.DataFrame([input_data])
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df = df[features]
    return df

def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    input_df = prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                             delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                             loan_purpose, loan_type)
    probability, credit_score, rating = calculate_credit_score(input_df)
    return probability, credit_score, rating

def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_
    default_probability = 1 / (1 + np.exp(-x))
    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability.flatten() * scale_length
    
    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'
    
    rating = get_rating(credit_score[0])
    return default_probability.flatten()[0], int(credit_score[0]), rating

def create_gauge_chart(value, title, min_val=0, max_val=100, threshold_good=70, threshold_warning=40):
    """Create a gauge chart for visualizing metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [0, threshold_warning], 'color': "#e74c3c"},
                {'range': [threshold_warning, threshold_good], 'color': "#f39c12"},
                {'range': [threshold_good, max_val], 'color': "#27ae60"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Main UI
# Header
st.markdown("""
    <div class="header-container">
        <div class="header-title">🏦 Credit Risk Assessment</div>
        <div class="header-subtitle">Advanced credit scoring and risk analysis platform powered by machine learning</div>
    </div>
""", unsafe_allow_html=True)

# Create two columns for layout
left_col, right_col = st.columns([2, 1])

with left_col:
    # Personal Information Section
    st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=18, step=1, max_value=100, value=28, help="Applicant's age in years")
    with col2:
        income = st.number_input('Annual Income (₹)', min_value=0, value=1200000, step=10000, help="Gross annual income")
    with col3:
        residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'], help="Current residence status")

    # Loan Details Section
    st.markdown('<div class="section-header">💰 Loan Details</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amount = st.number_input('Loan Amount (₹)', min_value=0, value=2560000, step=10000, help="Requested loan amount")
    with col2:
        loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36, help="Duration of loan in months")
    with col3:
        loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'], help="Type of loan")

    # Credit History Section
    st.markdown('<div class="section-header">📊 Credit History</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        credit_utilization_ratio = st.number_input('Credit Utilization Ratio (%)', min_value=0, max_value=100, step=1, value=30, help="Percentage of available credit being used")
    with col2:
        num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2, help="Number of currently active loans")
    with col3:
        loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'], help="Purpose of the loan")

    # Risk Indicators Section
    st.markdown('<div class="section-header">⚠️ Risk Indicators</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        delinquency_ratio = st.number_input('Delinquency Ratio (%)', min_value=0, max_value=100, step=1, value=30, help="Percentage of payments that are delinquent")
    with col2:
        avg_dpd_per_delinquency = st.number_input('Avg Days Past Due', min_value=0, value=20, step=1, help="Average days past due for delinquent accounts")

    # Calculate and display loan-to-income ratio
    loan_to_income_ratio = loan_amount / income if income > 0 else 0
    st.info(f"📈 **Loan to Income Ratio:** {loan_to_income_ratio:.2f}")

with right_col:
    st.markdown('<div class="section-header">📋 Quick Summary</div>', unsafe_allow_html=True)
    
    # Display summary metrics in cards
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Applicant Age</div>
            <div class="metric-value">{age} years</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Annual Income</div>
            <div class="metric-value">₹{income:,.0f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Loan Amount</div>
            <div class="metric-value">₹{loan_amount:,.0f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Loan Tenure</div>
            <div class="metric-value">{loan_tenure_months} months</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Credit Utilization</div>
            <div class="metric-value">{credit_utilization_ratio}%</div>
        </div>
    """, unsafe_allow_html=True)

# Calculate Risk Button
st.markdown("---")
if st.button('🔍 Calculate Credit Risk', use_container_width=True):
    try:
        # Get prediction
        probability, credit_score, rating = predict(
            age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type
        )
        
        # Display Results
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Risk Assessment Results")
        
        # Create three columns for results
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            # Gauge chart for credit score
            fig = create_gauge_chart(credit_score, "Credit Score", min_val=300, max_val=900, threshold_good=750, threshold_warning=650)
            st.plotly_chart(fig, use_container_width=True)
        
        with res_col2:
            st.markdown("#### 📈 Risk Metrics")
            # Display default probability as progress bar
            st.markdown("**Default Probability**")
            st.progress(probability)
            st.write(f"{probability:.2%}")
            
            # Risk Level
            st.markdown("**Risk Level**")
            if probability < 0.2:
                st.success("🟢 Low Risk")
            elif probability < 0.4:
                st.warning("🟡 Moderate Risk")
            elif probability < 0.6:
                st.warning("🟠 High Risk")
            else:
                st.error("🔴 Very High Risk")
        
        with res_col3:
            # Rating display
            rating_colors = {
                'Excellent': 'rating-excellent',
                'Good': 'rating-good',
                'Average': 'rating-average',
                'Poor': 'rating-poor'
            }
            rating_class = rating_colors.get(rating, 'rating-average')
            
            st.markdown("#### ⭐ Credit Rating")
            st.markdown(f"""
                <div style="text-align: center;">
                    <div class="rating-badge {rating_class}">
                        {rating}
                    </div>
                    <div style="margin-top: 1rem; font-size: 0.9rem; color: #7f8c8d;">
                        Credit Score: <strong>{credit_score}</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check all inputs and try again.")

# Footer
st.markdown("""
    <div class="footer">
        <p>© 2024 Lauki Finance | Powered by Machine Learning</p>
        <p style="font-size: 0.75rem;">This is a demonstration tool. Please consult with financial professionals for actual credit decisions.</p>
    </div>
""", unsafe_allow_html=True)