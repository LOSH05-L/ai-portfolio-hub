import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score

# 1. Page Configuration
st.set_page_config(
    page_title="AI Intelligence Hub", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Glossy Glassmorphism & Neon CSS Injection
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(3, 7, 18) 90%);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f8fafc;
    }

    /* Glossy Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 24px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }

    /* Glowing Headings */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    
    .gradient-title {
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.75rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    /* Custom Glossy Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 12px;
        padding: 12px 24px;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4);
    }

    /* Sliders & Interactive Elements */
    .stSlider {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    /* Metric Highlights */
    .metric-highlight {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .metric-highlight-green {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4ade80 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<p class="gradient-title">✨ AI Intelligence Hub</p>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Next-generation predictive modeling suite featuring real-time glassmorphic UI inference.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🍷 Wine Cultivar Classifier", "🏠 California Housing Estimator"])

# ==================== TAB 1: CLASSIFICATION MODEL ====================
with tab1:
    wine = load_wine()
    X_w = pd.DataFrame(wine.data, columns=wine.feature_names)
    y_w = wine.target
    
    X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(X_w, y_w, test_size=0.2, random_state=42)
    scaler_w = StandardScaler()
    X_w_train_scaled = scaler_w.fit_transform(X_w_train)
    X_w_test_scaled = scaler_w.transform(X_w_test)
    
    clf_model = RandomForestClassifier(random_state=42)
    clf_model.fit(X_w_train_scaled, y_w_train)
    
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎛️ Chemical Parameter Controls")
        st.write("Tune sliders to simulate laboratory chemical compositions.")
        user_inputs = {}
        for feature in wine.feature_names:
            min_val = float(X_w[feature].min())
            max_val = float(X_w[feature].max())
            mean_val = float(X_w[feature].mean())
            user_inputs[feature] = st.slider(f"{feature}", min_val, max_val, mean_val, key=f"wine_{feature}")
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Live Model Inference")
        
        input_df = pd.DataFrame([user_inputs])
        input_scaled = scaler_w.transform(input_df)
        prediction = clf_model.predict(input_scaled)[0]
        pred_class = wine.target_names[prediction]
        acc = accuracy_score(y_w_test, clf_model.predict(X_w_test_scaled)) * 100
        
        st.markdown('<p style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; margin-bottom: 0;">PREDICTED CULTIVAR</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-highlight">{pred_class.upper()}</div>', unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
        
        st.metric(label="Random Forest Accuracy", value=f"{acc:.2f}%", delta="Optimized Model")
        st.info("✨ **Interactive Feedback:** Adjusting feature sliders instantly triggers vector scaling and pipeline classification updates.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 2: REGRESSION MODEL ====================
with tab2:
    housing = fetch_california_housing(as_frame=True)
    X_h = housing.data
    y_h = housing.target
    
    X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
    scaler_h = StandardScaler()
    X_h_train_scaled = scaler_h.fit_transform(X_h_train)
    X_h_test_scaled = scaler_h.transform(X_h_test)
    
    reg_model = LinearRegression()
    reg_model.fit(X_h_train_scaled, y_h_train)
    
    col3, col4 = st.columns([1.2, 1], gap="large")
    
    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎛️ Neighborhood Demographics")
        st.write("Modify housing indicators to compute dynamic property values.")
        h_inputs = {}
        for feature in housing.feature_names:
            min_val = float(X_h[feature].min())
            max_val = float(X_h[feature].max())
            mean_val = float(X_h[feature].mean())
            h_inputs[feature] = st.slider(f"{feature}", min_val, max_val, mean_val, key=f"house_{feature}")
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col4:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Valuation Analytics")
        
        h_input_df = pd.DataFrame([h_inputs])
        h_input_scaled = scaler_h.transform(h_input_df)
        h_prediction = reg_model.predict(h_input_scaled)[0]
        estimated_price = h_prediction * 100000
        r2 = r2_score(y_h_test, reg_model.predict(X_h_test_scaled)) * 100
        
        st.markdown('<p style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; margin-bottom: 0;">ESTIMATED MARKET VALUE</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-highlight-green">${estimated_price:,.2f}</div>', unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
        
        st.metric(label="Linear Regression R² Score", value=f"{r2:.2f}%", delta="Baseline Model")
        st.info("✨ **Interactive Feedback:** Real-time regression estimation updates as geographic and income parameters change.")
        st.markdown('</div>', unsafe_allow_html=True)