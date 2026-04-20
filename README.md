# 🥇 GoldSense — AI-Powered Gold Price Predictor

> **📚 College Project** — Developed by **Vishal Mehta (UID: 25MCD10055)** and **Karan Singh (UID: 25MCD10020)**
> MCA Data Science | Chandigarh University

---

## 🧠 Project Overview

**GoldSense** is an end-to-end Machine Learning application that predicts gold prices using a **Deep LSTM (Long Short-Term Memory) Neural Network** fused with **FinBERT Natural Language Processing** for real-time financial news sentiment analysis.

The system ingests multi-variate macroeconomic data — gold futures, the US Dollar Index, S&P 500, crude oil, and interest rates — engineers advanced technical indicators, and feeds them into a stacked LSTM architecture to forecast gold prices up to **30 days into the future**.

The project is deployed as a fully interactive **Streamlit web dashboard** with live price feeds, dynamic charts, buy/sell signals, and a live news sentiment analyser.

---

## 👨‍💻 Developers

| Name |
|---|
| Vishal Mehta |
| Karan Singh |

**Programme:** MCA Data Science
**Institution:** Chandigarh University

---

## ✨ Key Features

- 📈 **LSTM Deep Learning** — Stacked 2-layer LSTM trained on 6,000+ days of market data
- 🤖 **FinBERT Sentiment** — Financial NLP model analyses live news headlines for fear/confidence scores
- 🌐 **Live Gold Price API** — Real-time gold prices via metal price APIs
- 💱 **Multi-Currency & Purity** — Supports USD / INR, 24K / 22K, per gram or per 10 grams
- 📅 **Custom Date Range** — Explore and analyse any historical period
- 🔮 **1–30 Day Forecast** — Adjustable future forecast horizon
- 📊 **Interactive Dashboard** — Correlation heatmaps, error analysis, residual plots, precision scatter
- 📰 **Live News Sentiment** — Type any headline and instantly get an AI-generated Fear Score
- 🐳 **Docker Ready** — Fully containerised for one-command deployment

---

## 🏗️ Architecture

```
Yahoo Finance API
      │
      ▼
┌─────────────────────────────────┐
│   Multi-Variate Data (5 assets) │
│  Gold · USD · S&P500 · Oil · IR │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│     FinBERT Sentiment Engine    │
│  ProsusAI/finbert (HuggingFace) │
│  → Adds Fear Score (0.0 – 1.0)  │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│      Feature Engineering        │
│  SMA-50 · Volatility-30d        │
│  → 8 final features total       │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│     MinMax Scaling (0 → 1)      │
│  60-day lookback window         │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│       Deep LSTM Model           │
│  Input(60×8)                    │
│  → LSTM(64, return_seq=True)    │
│  → Dropout(0.2)                 │
│  → LSTM(32, return_seq=False)   │
│  → Dropout(0.2)                 │
│  → Dense(1) → Gold Price        │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│    Streamlit Dashboard          │
│  Charts · Forecast · Signals    │
└─────────────────────────────────┘
```

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Source** | Yahoo Finance (via `yfinance` library) |
| **Date Range** | November 2000 – April 2026 |
| **Total Records** | 6,358 trading days |
| **Target Variable** | Gold closing price (USD per Troy Ounce) |

### Features Used

| Feature | Ticker | Description |
|---|---|---|
| Gold | `GC=F` | Gold futures — the prediction target |
| USD Index | `DX-Y.NYB` | US Dollar strength (inverse correlation with gold) |
| S&P 500 | `^GSPC` | US equities — risk-on/risk-off signal |
| Crude Oil | `CL=F` | Inflation proxy |
| Interest Rate | `^IRX` | 13-week Treasury Bill rate |
| Sentiment Score | FinBERT | Daily fear/confidence score from news (0.0–1.0) |
| Volatility 30d | Engineered | Annualised 30-day rolling standard deviation |
| SMA-50 | Engineered | 50-day Simple Moving Average (trend indicator) |

---

## 🤖 Model Details

### LSTM Architecture

```python
model = Sequential([
    Input(shape=(60, 8)),           # 60-day lookback, 8 features
    LSTM(64, return_sequences=True),# First LSTM layer — learns broad patterns
    Dropout(0.2),                   # Prevents overfitting
    LSTM(32, return_sequences=False),# Second LSTM layer — distils patterns
    Dropout(0.2),                   # Prevents overfitting
    Dense(1)                        # Single output — predicted gold price
])
```

### Training Configuration

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Epochs | 50 (max) |
| Batch Size | 32 |
| Validation Split | 10% of training data |
| Early Stopping | Patience = 10 epochs |
| Train / Test Split | 85% / 15% |
| Lookback Window | 60 days |

### Evaluation Metrics

| Metric | Description |
|---|---|
| **R² Score** | How well the model explains price variance (1.0 = perfect) |
| **RMSE** | Root Mean Squared Error in USD |
| **MAE** | Mean Absolute Error in USD |
| **MAPE** | Mean Absolute Percentage Error |
| **Directional Accuracy** | Did the model correctly predict UP or DOWN? (>50% = better than chance) |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow / Keras |
| NLP Model | `ProsusAI/finbert` (HuggingFace Transformers) |
| Data | `yfinance`, `pandas`, `numpy` |
| Visualisation | `plotly`, `matplotlib`, `seaborn` |
| Web App | `streamlit` |
| ML Utilities | `scikit-learn`, `joblib` |
| Containerisation | Docker |

---

## 🚀 How to Run the Project

### Option 1 — Run Locally (Recommended for Development)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/goldsense.git
cd goldsense
```

#### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Generate the Dataset & Train the Model

Run the Jupyter Notebook to download data, train the LSTM, and generate the processed CSV:

```bash
jupyter notebook goldsense.ipynb
```

Run all cells from top to bottom. This will:
- Download market data from Yahoo Finance
- Run FinBERT sentiment analysis
- Engineer features (SMA-50, Volatility-30d)
- Train the LSTM model
- Save `model/lstm_model.keras`, `model/scaler.pkl`, and `goldsense_processed.csv`

> ⚠️ **Note:** Training may take 5–15 minutes depending on your hardware.

#### Step 5: Launch the Streamlit App

```bash
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

---

### Option 2 — Run with Docker

#### Step 1: Build the Docker Image

```bash
docker build -t goldsense .
```

#### Step 2: Run the Container

```bash
docker run -p 8501:8501 goldsense
```

Then open your browser at: **http://localhost:8501**

---

### Option 3 — Quick Start (Skip Training)

If you already have the `goldsense_processed.csv` file and saved model files, you can skip the notebook entirely and just run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
goldsense/
│
├── app.py                      # Main Streamlit dashboard
├── goldsense.ipynb             # Jupyter Notebook — training pipeline
├── goldsense_processed.csv     # Pre-processed dataset (generated by notebook)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
│
├── model/
│   ├── lstm_model.keras        # Trained LSTM model (generated by notebook)
│   └── scaler.pkl              # Fitted MinMaxScaler (generated by notebook)
│
├── utils/
│   ├── data_loader.py          # Live gold price API fetcher
│   ├── predictor.py            # LSTM inference logic
│   ├── signals.py              # Buy/Sell/Hold signal generator
│   ├── metrics.py              # RMSE, MAE, R², Directional Accuracy
│   └── sentiment.py            # FinBERT pipeline wrapper
│
└── src/
    └── streamlit_app.py        # Alternative entry point
```

---

## 📊 Dashboard Tabs

| Tab | Content |
|---|---|
| **Price & Forecast** | Historical actual vs LSTM predicted prices · Future forecast line · Present-day divider |
| **Macro Correlation** | Seaborn heatmap · Plotly interactive heatmap · Normalised asset growth comparison |
| **Error Analysis** | Residual distribution histogram · Precision scatter plot · Performance summary metrics |
| **Sentiment** | FinBERT fear score vs 30-day market volatility overlay chart |

---

## 🌍 UN SDG Alignment

This project contributes to the following United Nations Sustainable Development Goals:

| SDG | Goal | How GoldSense contributes |
|---|---|---|
| **SDG 1** | No Poverty | Empowers small investors and people in developing nations to make smarter gold savings decisions |
| **SDG 8** | Decent Work & Economic Growth | Promotes informed financial activity and healthier markets |
| **SDG 9** | Industry, Innovation & Infrastructure | Uses state-of-the-art AI (LSTM + FinBERT) to modernise financial forecasting |
| **SDG 10** | Reduced Inequalities | Democratises institutional-grade market intelligence for everyday people |

---

## ⚠️ Disclaimer

> GoldSense is an academic research project built for educational purposes. All predictions and signals generated by this system are **not financial advice**. Do not make real investment decisions based solely on this tool. Always consult a qualified financial advisor before trading.

---

## 📄 License

This project is developed as part of an academic programme at **Chandigarh University** and is intended for educational use only.

---

<div align="center">

Made with ❤️ by **Vishal Mehta** & **Karan Singh**

MCA Data Science · Chandigarh University

</div>
