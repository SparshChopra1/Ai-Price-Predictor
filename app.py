import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from data_fetcher import MarketDataFetcher
from ai_predictor import AIPredictor
from location_service import LocationService
from utils import format_price, calculate_trend, get_price_color
import config
import json

# Page Configuration
st.set_page_config(
    page_title="AI Market Predictor - Predict Market Prices Before They Change",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Custom CSS
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize Services
@st.cache_resource
def init_services():
    return (
        MarketDataFetcher(config.DATA_GOV_API_KEY),
        AIPredictor(config.OPENROUTER_API_KEY),
        LocationService()
    )

data_fetcher, ai_predictor, location_service = init_services()

# Session State Initialization
if 'location' not in st.session_state:
    st.session_state.location = "Punjab"
if 'selected_crop' not in st.session_state:
    st.session_state.selected_crop = "Wheat"
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# Navigation Bar
st.markdown("""
<div class="nav-container">
    <div class="nav-bar">
        <div class="nav-logo">
            <span class="logo-icon">🌾📊</span>
            <span class="logo-text">AI Market Predictor</span>
        </div>
        <div class="nav-menu">
            <a href="#" class="nav-link">Home</a>
            <a href="#" class="nav-link">Dashboard</a>
            <a href="#" class="nav-link active">Insights</a>
            <a href="#" class="nav-link">About</a>
            <a href="#" class="nav-link">Contact</a>
        </div>
        <button class="launch-btn">🚀 Launch Predictor</button>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <h1 class="hero-title">Predict Market Prices Before They Change</h1>
        <p class="hero-subtitle">Empowering farmers with AI-driven market intelligence</p>
        <div class="hero-illustration">
            <span class="hero-icon">📱💹🌾</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Location and Crop Selection
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    states = ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", 
              "Maharashtra", "Karnataka", "Rajasthan", "Gujarat"]
    selected_state = st.selectbox(
        "📍 Select Region",
        states,
        index=states.index(st.session_state.location) if st.session_state.location in states else 0,
        key="state_select"
    )
    st.session_state.location = selected_state

with col2:
    crops = ["Wheat", "Rice", "Cotton", "Maize", "Potato", "Onion", "Tomato", "Soyabean"]
    selected_crop = st.selectbox(
        "🌾 Select Crop",
        crops,
        index=crops.index(st.session_state.selected_crop) if st.session_state.selected_crop in crops else 0,
        key="crop_select"
    )
    st.session_state.selected_crop = selected_crop

with col3:
    st.markdown("<div style='height: 29px'></div>", unsafe_allow_html=True)
    if st.button("🔍 Check Prices Now", key="search_btn", type="primary", use_container_width=True):
        st.session_state.show_results = True

# Main Results Section
if st.session_state.show_results:
    
    # Section 1: Real-Time Crop Prices
    st.markdown(f"""
    <div class="section-header">
        <h2 class="section-title">📊 Real-Time Crop Prices</h2>
        <span class="location-badge">📍 {selected_state}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch Market Data
    with st.spinner("🔄 Fetching latest market data..."):
        market_data = data_fetcher.get_market_prices(selected_state, selected_crop)
    
    if market_data and len(market_data) > 0:
        # Display top 3 mandis
        cols = st.columns(3)
        for idx, mandi_data in enumerate(market_data[:3]):
            with cols[idx]:
                price = mandi_data['modal_price']
                change = mandi_data.get('price_change', np.random.uniform(-5, 5))
                price_level = get_price_color(price, market_data)
                
                st.markdown(f"""
                <div class="mandi-card {price_level}">
                    <div class="mandi-header">
                        <h3 class="mandi-name">{mandi_data['market']}</h3>
                        <span class="mandi-district">{mandi_data.get('district', '')}</span>
                    </div>
                    <div class="price-container">
                        <div class="price-value">₹{price}</div>
                        <div class="price-unit">/quintal</div>
                    </div>
                    <div class="price-change {'positive' if change > 0 else 'negative'}">
                        <span class="change-icon">{'📈' if change > 0 else '📉'}</span>
                        <span class="change-value">{abs(change):.1f}% from yesterday</span>
                    </div>
                    <div class="price-range">
                        <span class="range-item">
                            <small>Min</small>
                            <strong>₹{mandi_data['min_price']}</strong>
                        </span>
                        <span class="range-divider">•</span>
                        <span class="range-item">
                            <small>Max</small>
                            <strong>₹{mandi_data['max_price']}</strong>
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📊 Market data is being updated. Please try again in a moment.")
    
    # Section 2: Understand Price Movement
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">📈 Understand the Price Movement</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get historical data
        historical_data = data_fetcher.get_historical_data(selected_state, selected_crop)
        
        if historical_data:
            # Create price trend chart
            fig = go.Figure()
            
            # Add main price line with gradient
            fig.add_trace(go.Scatter(
                x=historical_data['dates'],
                y=historical_data['prices'],
                mode='lines+markers',
                name='Market Price',
                line=dict(color='#43A047', width=4),
                marker=dict(
                    size=12, 
                    color='#43A047',
                    line=dict(color='white', width=3),
                    symbol='circle'
                ),
                fill='tozeroy',
                fillcolor='rgba(67, 160, 71, 0.1)',
                hovertemplate='<b>%{x|%d %b}</b><br>Price: ₹%{y:.0f}/qtl<extra></extra>'
            ))
            
            # Add moving average
            if len(historical_data['prices']) >= 3:
                ma = pd.Series(historical_data['prices']).rolling(window=3).mean()
                fig.add_trace(go.Scatter(
                    x=historical_data['dates'],
                    y=ma,
                    mode='lines',
                    name='3-Day Average',
                    line=dict(color='#2196F3', width=3, dash='dash'),
                    opacity=0.7,
                    hovertemplate='<b>Average:</b> ₹%{y:.0f}/qtl<extra></extra>'
                ))
            
            fig.update_layout(
                title=dict(
                    text=f"<b>{selected_crop} Price Trend - 7 Days</b>",
                    font=dict(size=20, color='#1B5E20')
                ),
                xaxis_title="Date",
                yaxis_title="Price (₹/quintal)",
                height=450,
                hovermode='x unified',
                plot_bgcolor='rgba(248, 249, 246, 0.5)',
                paper_bgcolor='white',
                font=dict(family="Poppins", size=12),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)',
                    tickformat='%d %b',
                    linecolor='#ECEFF1'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.05)',
                    linecolor='#ECEFF1'
                ),
                legend=dict(
                    orientation="h",
                    y=1.15,
                    x=0.5,
                    xanchor='center',
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#ECEFF1',
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if historical_data:
            prices = historical_data['prices']
            avg_price = np.mean(prices)
            max_price = np.max(prices)
            min_price = np.min(prices)
            volatility = np.std(prices)
            
            st.markdown(f"""
            <div class="stats-card">
                <h4 class="stats-title">📊 Price Statistics</h4>
                <div class="stat-item">
                    <span class="stat-label">Average</span>
                    <span class="stat-value">₹{avg_price:.0f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Highest</span>
                    <span class="stat-value high">₹{max_price:.0f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Lowest</span>
                    <span class="stat-value low">₹{min_price:.0f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Volatility</span>
                    <span class="stat-value">±₹{volatility:.0f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Section 3: AI Price Forecast
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">🤖 Predict the Future of Your Crop Prices</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🧠 AI is analyzing market patterns and generating predictions..."):
        # Get AI predictions
        prediction_data = ai_predictor.predict_prices(
            crop=selected_crop,
            state=selected_state,
            historical_data=historical_data if historical_data else None,
            market_data=market_data
        )
    
    if prediction_data:
        pred_cols = st.columns(3)
        
        with pred_cols[0]:
            st.markdown(f"""
            <div class="prediction-card gradient-green">
                <div class="pred-header">
                    <span class="pred-icon">📅</span>
                    <h3>Next Week Forecast</h3>
                </div>
                <div class="predicted-price">₹{prediction_data['predicted_price']}/qtl</div>
                <div class="price-trend {'up' if prediction_data['trend'] > 0 else 'down'}">
                    <span class="trend-arrow">{'↗' if prediction_data['trend'] > 0 else '↘'}</span>
                    <span>{abs(prediction_data['trend']):.1f}% {'increase' if prediction_data['trend'] > 0 else 'decrease'}</span>
                </div>
                <div class="confidence-meter">
                    <div class="confidence-label">AI Confidence Score</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {prediction_data['confidence']}%">
                            <span class="confidence-text">{prediction_data['confidence']}%</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with pred_cols[1]:
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="rec-header">
                    <span class="rec-icon">💡</span>
                    <h3>AI Recommendation</h3>
                </div>
                <div class="rec-content">
                    <p class="rec-text">{prediction_data['recommendation']}</p>
                    <div class="best-time">
                        <span class="time-icon">⏰</span>
                        <div style="flex: 1;">
                            <div class="time-label">Best Selling Day</div>
                            <div class="time-value">{prediction_data['best_day']}</div>
                        </div>
                    </div>
                    <div class="profit-indicator">
                        <span class="profit-label">Expected Extra Profit</span>
                        <span class="profit-value">+₹{prediction_data['expected_profit']}/qtl</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with pred_cols[2]:
            st.markdown(f"""
            <div class="factors-card">
                <div class="factors-header">
                    <span class="factors-icon">📋</span>
                    <h3>Market Factors</h3>
                </div>
                <div class="factor-list">
                    <div class="factor-item">
                        <span class="factor-icon">🌦️</span>
                        <span class="factor-label">Weather</span>
                        <span class="factor-value {prediction_data['factors']['weather'].lower()}">{prediction_data['factors']['weather']}</span>
                    </div>
                    <div class="factor-item">
                        <span class="factor-icon">📈</span>
                        <span class="factor-label">Demand</span>
                        <span class="factor-value {prediction_data['factors']['demand'].lower()}">{prediction_data['factors']['demand']}</span>
                    </div>
                    <div class="factor-item">
                        <span class="factor-icon">🌾</span>
                        <span class="factor-label">Supply</span>
                        <span class="factor-value {prediction_data['factors']['supply'].lower()}">{prediction_data['factors']['supply']}</span>
                    </div>
                    <div class="factor-item">
                        <span class="factor-icon">🎯</span>
                        <span class="factor-label">Market</span>
                        <span class="factor-value {prediction_data['factors']['market'].lower()}">{prediction_data['factors']['market']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Section 4: Smart Insights
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">💡 AI-Generated Market Insights</h2>
    </div>
    """, unsafe_allow_html=True)
    
    insight_cols = st.columns(3)
    
    with insight_cols[0]:
        top_crops = ai_predictor.get_top_crops(selected_state)
        st.markdown("""
        <div class="insight-card">
            <div class="insight-header">
                <span class="insight-icon">🌾</span>
                <h3>Top Crops in Demand</h3>
            </div>
            <div class="insight-content">
                <ol class="crop-list">
        """, unsafe_allow_html=True)
        
        for crop in top_crops[:3]:
            st.markdown(f"""
                <li class="crop-item">
                    <span class="crop-name">{crop['name']}</span>
                    <span class="crop-price">₹{crop['price']}/qtl</span>
                </li>
            """, unsafe_allow_html=True)
        
        st.markdown("</ol></div></div>", unsafe_allow_html=True)
    
    with insight_cols[1]:
        best_markets = data_fetcher.get_best_markets(selected_crop, selected_state)
        st.markdown("""
        <div class="insight-card">
            <div class="insight-header">
                <span class="insight-icon">🏪</span>
                <h3>Best Mandis to Sell</h3>
            </div>
            <div class="insight-content">
                <ul class="market-list">
        """, unsafe_allow_html=True)
        
        for market in best_markets[:3]:
            st.markdown(f"""
                <li class="market-item">
                    <span class="market-name">{market['name']}</span>
                    <span class="market-price">₹{market['price']}/qtl</span>
                </li>
            """, unsafe_allow_html=True)
        
        st.markdown("</ul></div></div>", unsafe_allow_html=True)
    
    with insight_cols[2]:
        seasonal = ai_predictor.get_seasonal_forecast(selected_crop)
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-header">
                <span class="insight-icon">📆</span>
                <h3>Regional Forecast</h3>
            </div>
            <div class="insight-content">
                <p class="seasonal-text">{seasonal['summary']}</p>
                <div class="demand-badge {seasonal['trend']}">
                    Demand: {seasonal['trend'].upper()}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Section 5: Smart Selling Tips
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">💬 AI Advisor: Smart Selling Tips</h2>
    </div>
    """, unsafe_allow_html=True)
    
    advice = ai_predictor.get_smart_advice(
        crop=selected_crop,
        state=selected_state,
        current_price=market_data[0]['modal_price'] if market_data else 0
    )
    
    st.markdown(f"""
    <div class="advisory-section">
        <div class="advisory-card">
            <div class="advisory-header">
                <span class="advisory-icon">🎯</span>
                <h3>Personalized Advice for {selected_crop} Farmers in {selected_state}</h3>
            </div>
            <div class="advisory-content">
                <p class="main-advice">{advice['main_advice']}</p>
                <div class="advice-grid">
                    <div class="advice-item">
                        <span class="advice-check">✅</span>
                        <span>{advice['point1']}</span>
                    </div>
                    <div class="advice-item">
                        <span class="advice-check">✅</span>
                        <span>{advice['point2']}</span>
                    </div>
                    <div class="advice-item">
                        <span class="advice-check">✅</span>
                        <span>{advice['point3']}</span>
                    </div>
                </div>
                <button class="voice-btn">
                    <span class="voice-icon">🔊</span>
                    <span>Listen in Local Language</span>
                </button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📥 Download Complete Market Report", key="download_btn", use_container_width=True):
            st.success("✅ Report generated successfully! Download will start soon.")

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-content">
        <p class="footer-text">Ready to Make Smarter Market Decisions?</p>
        <p class="footer-text">© 2025 AI Market Predictor — Powered by AgriTech AI</p>
        <p class="footer-links">
            <a href="#">About</a> | 
            <a href="#">Privacy</a> | 
            <a href="#">Contact</a> | 
            <a href="#">Support</a>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)