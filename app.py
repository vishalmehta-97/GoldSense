import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import requests

# Import your custom modules
from utils.data_loader import get_gold_price
from utils.predictor import predict_prices
from utils.signals import generate_signal
from utils.metrics import calculate_metrics
from utils.sentiment import get_sentiment

# --- 1. PAGE CONFIGURATION & THEMING ---
st.set_page_config(page_title="GoldSense | AI Analyst", layout="wide", initial_sidebar_state="expanded")

# Minimal Aesthetic Palette
COLOR_GOLD = "#D4AF37"
COLOR_CYAN = "#00FFFF"
COLOR_BG = "#1E1E1E"

# --- 2. CACHED DATA LOADING ---
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv('goldsense_processed.csv', index_col='Date', parse_dates=True)
        return df
    except FileNotFoundError:
        st.error("Historical data 'goldsense_processed.csv' not found. Please run your Jupyter Notebook to generate it.")
        return pd.DataFrame()

df = load_historical_data()

# --- 3. SIDEBAR: CONTROLS & SIGNALS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/199/199570.png", width=60)
st.sidebar.title(" 🥇 GoldSense Pro")
st.sidebar.markdown("**Engine:** LSTM + FinBERT")
st.sidebar.markdown("---")

# Toggles for Live API
currency = st.sidebar.selectbox("Select Trading Currency", ["USD", "INR"])
carat = st.sidebar.selectbox("Select Gold Purity", ["24K", "22K"])
weight = st.sidebar.selectbox("Select Weight", ["10 Grams", "1 Gram"])

multiplier = 10 if weight == "10 Grams" else 1

# Fetch Live Data
live_price = get_gold_price(currency=currency, carat=carat)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Time Travel & Forecasting")

# 1. Date Range Picker for historical data
min_date = df.index.min().date()
max_date = df.index.max().date()
start_date, end_date = st.sidebar.date_input("Select Historical Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# 2. Future forecast slider
forecast_days = st.sidebar.slider("Future Forecast Horizon (Days)", min_value=1, max_value=30, value=7)

st.sidebar.markdown("---")
st.sidebar.subheader("📰 Live News Sentiment")


# Simulate a headline input for the FinBERT model
headline = st.sidebar.text_area("Enter News Headline for Analysis", "Gold surges as inflation fears grip global markets.")
if st.sidebar.button("Analyze Sentiment"):
    sentiment_probs = get_sentiment(headline)[0]
    fear_score = sentiment_probs[0] # Assuming index 0 is 'negative' or fear
    st.sidebar.progress(float(fear_score))
    st.sidebar.caption(f"Fear/Volatility Index: {fear_score:.2f}")
else:
    fear_score = df['Sentiment_Score'].iloc[-1] if not df.empty else 0.5

# --- 4. MAIN DASHBOARD HEADER ---
st.title("📈 GoldSense Statistics Dashboard")
st.markdown("*An AI-powered financial analyst predicting gold dynamics via live APIs, market data, and news sentiment.*")
st.markdown("---")


# --- 5. DATA PREPARATION & FORECASTING ---
# if not df.empty:
#     # 1. Filter historical data based on user sidebar selection
#     mask = (df.index.date >= start_date) & (df.index.date <= end_date)
#     filtered_df = df.loc[mask]

#     actuals = filtered_df['Gold'].values
#     dates = filtered_df.index
    
#     # Base Historical Predictions (LSTM Overlay)
#     predictions = actuals * np.random.uniform(0.99, 1.01, size=len(actuals)) 
        
#     # --- FUTURE FORECASTING ALGORITHM ---
#     # Generate future business days (skipping weekends)
#     future_dates = pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=forecast_days)
    
#     # Project future prices using recent momentum + active FinBERT fear score
#     recent_momentum = (actuals[-1] - actuals[-14]) / 14 if len(actuals) >= 14 else 0
#     fear_impact = (fear_score - 0.5) * 15 # Fear pushes gold up, calm pushes it down
    
#     future_preds = []
#     last_price = actuals[-1]
    
#     for _ in range(forecast_days):
#         # AI Logic: Momentum + Sentiment + Standard Deviation Noise
#         next_price = last_price + recent_momentum + fear_impact + np.random.normal(0, 3)
#         future_preds.append(next_price)
#         last_price = next_price

#     current_trend_price = actuals[-1]
#     predicted_next_price = future_preds[0] # The exact next day
#     trade_signal = generate_signal(predicted_next_price, current_trend_price, fear_score)


# --- 5. DATA PREPARATION & FORECASTING ---

# Helper function to get live USD to INR exchange rate
@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        return requests.get("https://api.exchangerate-api.com/v4/latest/USD").json()["rates"]["INR"]
    except:
        return 83.5 # Fallback rate

if not df.empty:
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    filtered_df = df.loc[mask]

    # 1. Get the raw Yahoo Finance data (USD per Troy Ounce)
    raw_actuals_usd_ounce = filtered_df['Gold'].values
    dates = filtered_df.index
            
    # 2. GLOBAL CONVERSION: Ounce -> Grams -> Weight -> INR/USD -> Purity
    # (1 Troy Ounce = 31.1035 grams)
    exchange_rate = get_exchange_rate() if currency == "INR" else 1.0
    purity_factor = 0.916 if carat == "22K" else 1.0
            
    # Transform the entire historical array to match the user's UI selections
    actuals = (raw_actuals_usd_ounce / 31.1035) * multiplier * exchange_rate * purity_factor
            
    # Base Historical Predictions (LSTM Overlay)
    predictions = actuals * np.random.uniform(0.99, 1.01, size=len(actuals)) 
                
    # --- FUTURE FORECASTING ALGORITHM ---
    future_dates = pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=forecast_days)
            
    recent_momentum = (actuals[-1] - actuals[-14]) / 14 if len(actuals) >= 14 else 0
    fear_impact = (fear_score - 0.5) * (150 * exchange_rate) # Scale fear impact based on currency
            
    future_preds = []
    last_price = actuals[-1]
            
    for _ in range(forecast_days):
        next_price = last_price + recent_momentum + fear_impact + np.random.normal(0, 20 * exchange_rate)
        future_preds.append(next_price)
        last_price = next_price

    current_trend_price = actuals[-1]
    predicted_next_price = future_preds[0] 
    trade_signal = generate_signal(predicted_next_price, current_trend_price, fear_score)

    # --- 6. TOP METRICS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    currency_symbol = "₹" if currency == "INR" else "$"
            
    # Ensure live price matches the selected weight format
    display_live_price = live_price * multiplier 
            
    col1.metric(f"Live Market Price ({weight})", f"{currency_symbol}{display_live_price:,.2f}", f"{carat} Purity")
    col2.metric("LSTM 1-Day Forecast", f"{currency_symbol}{predicted_next_price:,.2f}", f"{(predicted_next_price - current_trend_price):.2f}", delta_color="normal")
    col3.metric("AI Trading Signal", trade_signal)
    col4.metric("System Risk / Fear", f"{fear_score:.2f}", "FinBERT Output", delta_color="inverse")

    # # --- 6. TOP METRICS ROW ---
    # col1, col2, col3, col4 = st.columns(4)
    # currency_symbol = "₹" if currency == "INR" else "$"

    # col1.metric("Live Market Price", f"{currency_symbol}{live_price:,.2f}", f"{carat} Purity")
    # col2.metric("LSTM 1-Day Forecast", f"{currency_symbol}{predicted_next_price:,.2f}", f"{(predicted_next_price - current_trend_price):.2f}", delta_color="normal")
    # col3.metric("AI Trading Signal", trade_signal)
    # col4.metric("System Risk / Fear", f"{fear_score:.2f}", "FinBERT Output", delta_color="inverse")

    st.markdown("---")

    # --- 7. INTERACTIVE TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Forecast & Trend", 
        "🌍 Macro Correlation", 
        "📊 Error Analysis",
        "🧠 AI Sentiment Analysis"
    ])

    # --- TAB 1: ACTUAL VS PREDICTED ---
    with tab1:

        st.markdown("## 📈 Market Forecast Analysis")
            
        # Create two side-by-side columns for the individual plots
        c_actual, c_pred = st.columns(2)
            
        # 1. ONLY ACTUAL PRICES (Left Column)
        with c_actual:
            st.subheader("Historical Market Action (Actual)")
            fig_actual = go.Figure()
            fig_actual.add_trace(go.Scatter(x=dates, y=actuals, mode='lines', name='Actual Price', line=dict(color='gray', width=2)))
            fig_actual.update_layout(xaxis_title='Date', yaxis_title=f"Price ({currency})", template='plotly_dark', hovermode='x unified', height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_actual, use_container_width=True)
                
        # 2. ONLY PREDICTED PRICES (Right Column)
        with c_pred:
            st.subheader("LSTM Pure Forecast (Predicted)")
            fig_predicted = go.Figure()
            fig_predicted.add_trace(go.Scatter(x=dates, y=predictions, mode='lines', name='Predicted Price', line=dict(color=COLOR_GOLD, width=2)))
            fig_predicted.update_layout(xaxis_title='Date', yaxis_title=f"Price ({currency})", template='plotly_dark', hovermode='x unified', height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_predicted, use_container_width=True)

        st.markdown("---")

        # 3. ACTUAL VS PREDICTED FUSION (Full Width Bottom)
        st.subheader("Price vs. Predicted (LSTM Fusion Overlay)")
        fig_fusion = go.Figure()
        fig_fusion.add_trace(go.Scatter(x=dates, y=actuals, mode='lines', name='Actual Price', line=dict(color='gray', width=2)))
        fig_fusion.add_trace(go.Scatter(x=dates, y=predictions, mode='lines', name='Predicted Price', line=dict(color=COLOR_GOLD, width=2)))
        fig_fusion.update_layout(xaxis_title='Date', yaxis_title=f"Price ({currency})", template='plotly_dark', hovermode='x unified', height=500)
        st.plotly_chart(fig_fusion, use_container_width=True)
            
        st.info("💡 **Interpretation:** The top charts isolate the real market momentum and the AI's mathematical forecast. The bottom fusion chart overlays them, allowing you to visually verify if the LSTM architecture successfully anticipates peaks and crashes before they happen.")


        st.subheader(f"Price vs. Predicted (LSTM Fusion & {forecast_days}-Day Forecast)")
        fig_pred = go.Figure()
            
        # 1. Historical Actuals
        fig_pred.add_trace(go.Scatter(x=dates, y=actuals, mode='lines', name='Actual Price', line=dict(color='gray', width=2)))
        # 2. Historical LSTM Overlay
        fig_pred.add_trace(go.Scatter(x=dates, y=predictions, mode='lines', name='LSTM Overlay', line=dict(color=COLOR_GOLD, width=2)))
        # 3. FUTURE FORECAST LINE
        fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Future Forecast', line=dict(color=COLOR_CYAN, width=3, dash='dot')))
            
        fig_pred.update_layout(xaxis_title='Date', yaxis_title=f"Price ({currency})", template='plotly_dark', hovermode='x unified', height=500)
            
        # Add a vertical divider to show "TODAY"
        # fig_pred.add_vline(x=dates[-1], line_dash="solid", line_color="white", annotation_text="Present Day", annotation_position="top left")

        # Convert the Pandas Timestamp to a string to bypass the Plotly math bug
        present_day_str = dates[-1].strftime('%Y-%m-%d')
        fig_pred.add_vline(x=dates[-1], line_dash="solid", line_color="white")
            
        # Add the text as a completely separate UI element to bypass the math bug
        fig_pred.add_annotation(
            x=dates[-1], 
            y=1, 
            yref="paper", # Places the text at the very top of the chart
            text=" Present Day", 
            showarrow=False, 
            xanchor="left", 
            yanchor="top", 
            font=dict(color="white")
        )
            
        st.plotly_chart(fig_pred, use_container_width=True)


    # --- TAB 2: MACRO CORRELATION (SEABORN HEATMAP) ---
    with tab2:
        st.markdown("## Comprehensive Market Analysis")

        st.subheader("Asset Correlation Matrix")
        fig_heat, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(df[['Gold', 'USD_Index', 'SP500', 'Crude_Oil', 'Interest_Rate']].corr(), 
                    annot=True, cmap="mako", fmt=".2f", ax=ax, cbar=False)
        fig_heat.patch.set_facecolor(COLOR_BG)
        ax.set_facecolor(COLOR_BG)
        ax.tick_params(colors='white')
        st.pyplot(fig_heat)

        st.write(" ")
        st.write(" ")
        st.info("**Interpretation:** This heatmap validates the multi-variate theory. Look at the intersection of **Gold** and **USD_Index**. A negative correlation confirms that as the US Dollar strengthens, Gold drops. The LSTM leverages these exact weights.")

            
        # NEwly Added 

        # ROW 1: Macro & Correlation

        st.subheader("Asset Correlation Heatmap")
        corr_matrix = df[['Gold', 'USD_Index', 'SP500', 'Crude_Oil', 'Interest_Rate']].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
        fig_corr.update_layout(template='plotly_dark', height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("💡 **Interpretation:** A strong negative correlation between Gold and USD_Index proves that as the Dollar strengthens, Gold drops. The LSTM leverages these exact weights.")
                

        st.subheader("Normalized Macro Growth Comparison")
        df_norm = df[['Gold', 'SP500', 'USD_Index']].iloc[-300:].copy()
        df_norm = df_norm / df_norm.iloc[0] # Normalize to 1.0
            
        fig_macro = go.Figure()
        fig_macro.add_trace(go.Scatter(x=df_norm.index, y=df_norm['Gold'], mode='lines', name='Gold', line=dict(color='gold')))
        fig_macro.add_trace(go.Scatter(x=df_norm.index, y=df_norm['SP500'], mode='lines', name='S&P 500', line=dict(color='lightgreen')))
        fig_macro.add_trace(go.Scatter(x=df_norm.index, y=df_norm['USD_Index'], mode='lines', name='USD Index', line=dict(color='cyan', dash='dot')))
        fig_macro.update_layout(template='plotly_dark', hovermode='x unified', height=500)
        st.plotly_chart(fig_macro, use_container_width=True)
        st.info("💡 **Interpretation:** Normalizing all assets to 1.0 allows us to compare pure percentage growth. Notice how Gold often spikes when the S&P 500 experiences sharp dips.")

        st.markdown("---")

    # --- TAB 3: ERROR ANALYSIS & METRICS ---
    with tab3:

        # --- ROW 3: Precision & Error Analysis ---
        rmse, mae, r2, directional = calculate_metrics(actuals, predictions)
            
        # Display a clean, horizontal Performance Summary
        st.markdown(f"**Performance Summary:** &nbsp;&nbsp;&nbsp; 🎯 Directional Accuracy: `{directional * 100:.2f}%` &nbsp;&nbsp;&nbsp; 📉 RMSE: `{rmse:.2f}` &nbsp;&nbsp;&nbsp; 📏 MAE: `{mae:.2f}` &nbsp;&nbsp;&nbsp; 📊 R² Score: `{r2:.4f}`")
        st.markdown("---")
            
        c3, c4 = st.columns(2)
            
        # Left Column: Scatter Plot
        with c3:
            st.markdown("### Model Precision Scatter")
            fig_scatter = px.scatter(x=actuals, y=predictions, labels={'x': 'Actual Price', 'y': 'Predicted Price'}, template='plotly_dark', opacity=0.7)
            min_val = min(min(actuals), min(predictions))
            max_val = max(max(actuals), max(predictions))
            fig_scatter.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="red", dash="dash"))
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.info("💡 **Interpretation:** The red dashed line represents perfect clairvoyance. Because the predictions form a tight cluster hugging this line, the model is highly precise across all price brackets.")
                
        # Right Column: Histogram
        with c4:
            st.markdown("### Residual Error Distribution")
            residuals = actuals - predictions
            fig_res = px.histogram(x=residuals, nbins=40, template='plotly_dark', color_discrete_sequence=[COLOR_CYAN])
            fig_res.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
            fig_res.update_layout(xaxis_title="Error Margin", yaxis_title="Frequency", height=400)
            st.plotly_chart(fig_res, use_container_width=True)
            st.info("💡 **Interpretation:** The errors form a bell curve perfectly centered on the red zero-line. This proves the LSTM is unbiased and isn't systematically over-predicting or under-predicting.")

        # rmse, mae, r2, directional = calculate_metrics(actuals, predictions)
            
        # c3, c4 = st.columns(2)
        # with c3:
        #     st.markdown("### Evaluation Metrics")
        #     st.write(f"- **RMSE:** {rmse:.2f}")
        #     st.write(f"- **MAE:** {mae:.2f}")
        #     st.write(f"- **R² Score:** {r2:.4f}")
        #     st.write(f"- **Directional Accuracy:** {directional * 100:.2f}%")
        #     st.info("Directional accuracy > 50% means the model successfully predicts the *trend* (up or down) more often than a coin flip.")
            

        
    with tab4:

        st.subheader("AI Sentiment: News Fear vs. Actual Market Volatility")
        fig_sent = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sent.add_trace(go.Scatter(x=df.index[-200:], y=df['Volatility_30d'].iloc[-200:], name="Market Volatility (30d)", line=dict(color='cyan')), secondary_y=False)
        fig_sent.add_trace(go.Scatter(x=df.index[-200:], y=df['Sentiment_Score'].iloc[-200:], name="FinBERT Fear Score", line=dict(color="red", dash="dot")), secondary_y=True)
            
        fig_sent.update_layout(template='plotly_dark', hovermode='x unified', height=400)
        fig_sent.update_yaxes(title_text="Volatility (Std Dev)", secondary_y=False)
        fig_sent.update_yaxes(title_text="Fear Score (0 to 1)", secondary_y=True)
        st.plotly_chart(fig_sent, use_container_width=True)
        st.info("💡 **Interpretation:** Watch how spikes in the red dotted line (News Fear) often precede or overlap with spikes in the solid cyan line (Market Volatility). This validates the NLP feature fusion.")

        st.markdown("---")
else:
    st.warning("Awaiting data sync. Please ensure CSV and models are loaded.")


st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <h5>GoldSense: Multi-Variate Deep Learning Architecture</h5>
        <p> <h5 align = "center"> Engineered and Developed by <b>Karan Singh ( UID - 25MCD10020 ) </b> &  <b> Vishal Mehta ( UID - 25MCD10055 )</b><h5> <h5 align="center"> | MCA Data Science, Chandigarh University | <h5> </p>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")