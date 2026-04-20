# utils/data_loader.py

import requests
import time

# -----------------------------
# CONFIG
# -----------------------------
FX_API = "https://api.exchangerate-api.com/v4/latest/USD"

# Cache to avoid excessive API calls
CACHE = {
    "gold_price": None,
    "timestamp": 0
}

CACHE_DURATION = 60  # seconds

# -----------------------------
# GET LIVE GOLD PRICE (Yahoo Finance)
# -----------------------------
def fetch_gold_price_usd():
    # Attempt 1: Yahoo Finance API directly
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            price = data['chart']['result'][0]['meta']['regularMarketPrice']
            if price:
                return price
    except Exception as e:
        print("Yahoo API Error:", e)

    # Attempt 2: yfinance fallback if installed
    try:
        import yfinance as yf
        ticker = yf.Ticker("GC=F")
        price = ticker.fast_info['lastPrice']
        return price
    except:
        pass

    return None

# -----------------------------
# GET USD → INR RATE
# -----------------------------
def fetch_usd_to_inr():
    try:
        response = requests.get(FX_API, timeout=5)
        response.raise_for_status()
        data = response.json()

        return data["rates"]["INR"]

    except Exception as e:
        print("FX API Error:", e)
        return None

def get_gold_price(currency="USD", carat="24K"):

    price_usd_ounce = fetch_gold_price_usd()

    # Fallback if APIs fail (Realistic 2024+ proxy)
    if price_usd_ounce is None:
        print("Using fallback price...")
        price_usd_ounce = 2350.0  

    price_usd_gram = price_usd_ounce / 31.1035

    if carat == "22K":
        price_usd_gram *= 0.916

    if currency == "INR":
        rate = fetch_usd_to_inr()

        if rate is None:
            print("Using fallback INR rate...")
            rate = 83.5  

        return round(price_usd_gram * rate, 2)

    return round(price_usd_gram, 2)