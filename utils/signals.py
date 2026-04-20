# utils/signals.py

def generate_signal(predicted, current, sentiment_score):
    
    if predicted > current and sentiment_score > 0.6:
        return "BUY 🟢"
    
    elif predicted < current and sentiment_score < 0.4:
        return "SELL 🔴"
    
    else:
        return "HOLD 🟡"