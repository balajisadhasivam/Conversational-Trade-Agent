import pandas as pd
import random
from datetime import datetime, timedelta

ASSETS = [
    'BTC', 'ETH', 'AAPL', 'TSLA', 'GOOG', 'AMZN', 'DOGE', 'SHIB', 'SOL', 'PEPE'
]
TRADE_TYPES = ['Buy', 'Sell']
TRADE_SOURCES = ['trend', 'news', 'meme', 'technical', 'value', 'sentiment']


def generate_synthetic_trades(num_trades=50, start_date='2023-01-01', out_path=None):
    trades = []
    start = datetime.strptime(start_date, '%Y-%m-%d')
    for i in range(1, num_trades + 1):
        date = start + timedelta(days=random.randint(0, 180))
        asset = random.choice(ASSETS)
        trade_type = random.choice(TRADE_TYPES)
        quantity = round(random.uniform(1, 100), 2)
        price = round(random.uniform(10, 50000), 2)
        profit_loss = round(random.uniform(-1000, 2000), 2)
        holding_period = random.randint(1, 30) if trade_type == 'Sell' else 0
        source = random.choices(
            TRADE_SOURCES,
            weights=[2 if asset in ['DOGE', 'SHIB', 'PEPE'] else 1 for _ in TRADE_SOURCES],
            k=1
        )[0]
        sentiment_score = round(random.uniform(-1, 1), 2)  # -1 (very negative) to 1 (very positive)
        trades.append({
            'Trade ID': f'TR-{i:04d}',
            'Date': date.strftime('%Y-%m-%d'),
            'Asset': asset,
            'Type': trade_type,
            'Quantity': quantity,
            'Price': price,
            'Profit/Loss': profit_loss,
            'Holding Period': holding_period,
            'Source': source,
            'Sentiment': sentiment_score
        })
    df = pd.DataFrame(trades)
    if out_path:
        df.to_csv(out_path, index=False)
    return df


def load_trades(path):
    return pd.read_csv(path) 