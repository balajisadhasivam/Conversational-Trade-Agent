import random
import pandas as pd

TRADER_STYLES = [
    'Day Trader',
    'Swing Trader',
    'Scalper',
    'Position Trader',
    'Algorithmic Trader',
    'Meme Trend Rider',
    'Sentiment Follower',
    'Value Investor'
]
RISK_APPETITES = ['Low', 'Medium', 'High']
EXPERIENCE_LEVELS = ['Novice', 'Intermediate', 'Expert']

class TraderPersona:
    def __init__(self, name='Alex', style=None, risk=None, experience=None, strategy_summary=None, preferences=None):
        self.name = name
        self.style = style or random.choice(TRADER_STYLES)
        self.risk = risk or random.choice(RISK_APPETITES)
        self.experience = experience or random.choice(EXPERIENCE_LEVELS)
        self.strategy_summary = strategy_summary or ""
        self.preferences = preferences or ""

    def describe(self):
        return (
            f"Trader Name: {self.name}\n"
            f"Style: {self.style}\n"
            f"Risk Appetite: {self.risk}\n"
            f"Experience Level: {self.experience}\n"
            f"Strategy: {self.strategy_summary}\n"
            f"Preferences: {self.preferences}"
        )

def assign_persona_from_trades(trades: pd.DataFrame, name='Alex'):
    # Analyze trades to assign persona
    avg_holding = trades['Holding Period'].mean()
    meme_trades = trades[trades['Source'] == 'meme']
    sentiment_trades = trades[trades['Source'] == 'sentiment']
    bluechip_trades = trades[trades['Asset'].isin(['BTC', 'ETH', 'AAPL', 'TSLA', 'GOOG', 'AMZN'])]
    meme_ratio = len(meme_trades) / len(trades)
    sentiment_ratio = len(sentiment_trades) / len(trades)
    bluechip_ratio = len(bluechip_trades) / len(trades)
    avg_sentiment = trades['Sentiment'].mean()
    avg_profit = trades['Profit/Loss'].mean()
    risk = 'High' if trades['Profit/Loss'].std() > 800 else 'Low' if trades['Profit/Loss'].std() < 400 else 'Medium'
    experience = 'Expert' if len(trades) > 100 else 'Intermediate' if len(trades) > 50 else 'Novice'

    # Assign style and strategy
    if meme_ratio > 0.25:
        style = 'Meme Trend Rider'
        strategy_summary = "I chase meme coins and ride trends early, aiming for quick profits."
        preferences = "I prefer meme tokens like DOGE, SHIB, PEPE and act on social sentiment."
    elif sentiment_ratio > 0.2:
        style = 'Sentiment Follower'
        strategy_summary = "I follow market sentiment and trade based on social/news signals."
        preferences = "I look for positive sentiment spikes and trade accordingly."
    elif avg_holding > 10 and bluechip_ratio > 0.5:
        style = 'Value Investor'
        strategy_summary = "I invest in blue-chip assets for the long term, focusing on value."
        preferences = "I prefer assets like BTC, ETH, AAPL, TSLA, GOOG, AMZN."
    elif avg_holding < 3:
        style = 'Scalper'
        strategy_summary = "I make quick trades, holding positions for a very short time."
        preferences = "I look for small, frequent profits."
    elif avg_holding < 7:
        style = 'Day Trader'
        strategy_summary = "I open and close trades within a few days, capitalizing on short-term moves."
        preferences = "I prefer liquid assets and technical setups."
    else:
        style = 'Swing Trader'
        strategy_summary = "I hold trades for several days to weeks, riding medium-term trends."
        preferences = "I look for strong trends and technical patterns."

    return TraderPersona(
        name=name,
        style=style,
        risk=risk,
        experience=experience,
        strategy_summary=strategy_summary,
        preferences=preferences
    ) 