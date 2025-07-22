import pandas as pd

def load_wallet_transactions(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    if 'block_time' in df.columns:
        df['block_time'] = pd.to_datetime(df['block_time'], errors='coerce')
    return df

def load_wallet_portfolio(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    return df

def load_token_feed(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df
