import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🏡 AI House Price Predictor Pro",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODERN STYLING
# ============================================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .main-header {
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
            backdrop-filter: blur(20px);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            text-align: center;
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        .main-header h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
            margin: 0;
        }
        
        .main-header p {
            color: #64748b;
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }
        
        .card {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(102,126,234,0.4);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        div.stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(102,126,234,0.4);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102,126,234,0.6);
        }
        
        .prediction-result {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 15px 40px rgba(102,126,234,0.5);
            margin: 2rem 0;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255,255,255,0.9);
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .info-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .success-box {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DIRECTORY SETUP
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CSV_PATH = os.path.join(DATA_DIR, "california_housing.csv")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.pkl")

# ============================================================================
# DATA & MODEL MANAGEMENT
# ============================================================================


@st.cache_data
def load_data():
    """Load California housing dataset"""
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    else:
        housing = fetch_california_housing(as_frame=True)
        data = housing.frame
        data.to_csv(CSV_PATH, index=False)
        return data


@st.cache_resource
def train_model():
    """Train the prediction model"""
    data = load_data()

    X = data.drop("MedHouseVal", axis=1)
    y = data["MedHouseVal"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Calculate metrics
    y_pred = model.predict(X_test)
    metrics = {
        'r2_score': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'train_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(metrics, METRICS_PATH)

    return model, scaler, metrics


@st.cache_resource
def load_model():
    """Load or train model"""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        metrics = joblib.load(METRICS_PATH) if os.path.exists(
            METRICS_PATH) else {}
        return model, scaler, metrics
    else:
        return train_model()

# ============================================================================
# MAIN APPLICATION
# ============================================================================


# Header
st.markdown("""
    <div class="main-header">
        <h1>🏡 AI House Price Predictor Pro</h1>
        <p>Advanced California Housing Market Analysis | Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings & Info")

    if st.button("🔄 Retrain Model"):
        with st.spinner("Retraining model..."):
            st.cache_resource.clear()
            model, scaler, metrics = train_model()
            st.success("✅ Model retrained successfully!")

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    # Load model and metrics
    model, scaler, metrics = load_model()

    if metrics:
        st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
        st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
        st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
        st.caption(f"Last trained: {metrics.get('train_date', 'Unknown')}")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    This application uses Random Forest regression to predict California house prices based on:
    - Location (Latitude/Longitude)
    - Demographics
    - House characteristics
    """)

# Main content tabs
tab1, tab2, tab3 = st.tabs(
    ["🏠 Price Prediction", "📈 Data Insights", "📋 About Dataset"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📝 Enter Property Details")

        input_col1, input_col2 = st.columns(2)

        with input_col1:
            MedInc = st.slider(
                "💰 Median Income (in $10k)",
                min_value=0.5,
                max_value=15.0,
                value=5.0,
                step=0.1,
                help="Median household income in the block group"
            )

            HouseAge = st.slider(
                "🏚️ House Age (years)",
                min_value=1,
                max_value=52,
                value=20,
                help="Median age of houses in the block"
            )

            AveRooms = st.slider(
                "🚪 Average Rooms",
                min_value=1.0,
                max_value=15.0,
                value=6.0,
                step=0.1,
                help="Average number of rooms per household"
            )

            AveBedrms = st.slider(
                "🛏️ Average Bedrooms",
                min_value=0.5,
                max_value=6.0,
                value=1.0,
                step=0.1,
                help="Average number of bedrooms per household"
            )

        with input_col2:
            Population = st.slider(
                "👥 Population",
                min_value=100,
                max_value=5000,
                value=1000,
                step=50,
                help="Total population in the block"
            )

            AveOccup = st.slider(
                "🏘️ Avg Occupancy",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.1,
                help="Average number of household members"
            )

            Latitude = st.slider(
                "📍 Latitude",
                min_value=32.0,
                max_value=42.0,
                value=36.0,
                step=0.1
            )

            Longitude = st.slider(
                "📍 Longitude",
                min_value=-124.0,
                max_value=-114.0,
                value=-120.0,
                step=0.1
            )

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔮 PREDICT HOUSE PRICE"):
            input_data = pd.DataFrame([{
                "MedInc": MedInc,
                "HouseAge": HouseAge,
                "AveRooms": AveRooms,
                "AveBedrms": AveBedrms,
                "Population": Population,
                "AveOccup": AveOccup,
                "Latitude": Latitude,
                "Longitude": Longitude
            }])

            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]
            price = prediction * 100000

            # Additional cost estimates
            furniture_cost = price * 0.08
            interior_cost = price * 0.05
            renovation_cost = price * 0.12
            total_cost = price + furniture_cost + interior_cost

            st.markdown(f"""
                <div class="prediction-result">
                    <h2>🏠 Estimated House Price</h2>
                    <div class="metric-value">${price:,.2f}</div>
                    <p style="margin-top: 1rem; opacity: 0.9;">Based on AI analysis of property characteristics</p>
                </div>
            """, unsafe_allow_html=True)

            # Cost breakdown
            breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)

            with breakdown_col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Furniture</div>
                        <div class="metric-value">${furniture_cost:,.0f}</div>
                    </div>
                """, unsafe_allow_html=True)

            with breakdown_col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Interior Design</div>
                        <div class="metric-value">${interior_cost:,.0f}</div>
                    </div>
                """, unsafe_allow_html=True)

            with breakdown_col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Renovation Budget</div>
                        <div class="metric-value">${renovation_cost:,.0f}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class="success-box">
                    <h3 style="margin:0;">💰 Total Investment Estimate: ${total_cost:,.2f}</h3>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Quick Tips")
        st.markdown("""
        **Factors that increase value:**
        - Higher median income
        - Optimal room-to-bedroom ratio
        - Prime locations (coastal areas)
        - Lower population density
        
        **Factors that decrease value:**
        - Very old houses (40+ years)
        - High occupancy rates
        - Inland locations
        - Overcrowding
        """)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Dataset Overview")

    data = load_data()

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.metric("Total Samples", f"{len(data):,}")
        st.metric("Features", len(data.columns) - 1)

    with viz_col2:
        st.metric("Avg House Price",
                  f"${data['MedHouseVal'].mean() * 100000:,.0f}")
        st.metric(
            "Price Range", f"${data['MedHouseVal'].min() * 100000:,.0f} - ${data['MedHouseVal'].max() * 100000:,.0f}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Visualizations
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Price Distribution")

    fig = px.histogram(
        data,
        x='MedHouseVal',
        nbins=50,
        title='Distribution of House Prices',
        labels={'MedHouseVal': 'House Price (in $100k)'},
        color_discrete_sequence=['#667eea']
    )
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Geographic Price Distribution")

    fig2 = px.scatter(
        data.sample(5000),
        x='Longitude',
        y='Latitude',
        color='MedHouseVal',
        size='Population',
        title='California Housing Prices by Location',
        color_continuous_scale='Viridis',
        labels={'MedHouseVal': 'Price ($100k)'}
    )
    fig2.update_layout(template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📋 California Housing Dataset")

    st.markdown("""
    The California Housing dataset contains information from the 1990 California census.
    
    **Features:**
    - **MedInc**: Median income in block group
    - **HouseAge**: Median house age in block group
    - **AveRooms**: Average number of rooms per household
    - **AveBedrms**: Average number of bedrooms per household
    - **Population**: Block group population
    - **AveOccup**: Average number of household members
    - **Latitude**: Block group latitude
    - **Longitude**: Block group longitude
    
    **Target:**
    - **MedHouseVal**: Median house value (in $100,000s)
    """)

    st.markdown("### 📊 Sample Data")
    st.dataframe(load_data().head(10), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: white; padding: 2rem;">
        <h4 style="color: white;">🚀 Developed by Dheeraj Muley</h4>
        <p style="color: rgba(255,255,255,0.8);">
            Powered by Streamlit • Scikit-learn • Plotly<br>
            © 2026 All Rights Reserved
        </p>
    </div>
""", unsafe_allow_html=True)
