import yfinance as yf
import pandas as pd
import numpy as np
import ta

start="2000-01-01"
end="2025-04-29"

tickers = ["SKF-B.ST", "NDA-SE.ST", "ASSA-B.ST",
 "ABB.ST", "SWED-A.ST", "TEL2-B.ST", "EVO.ST",
 "SAAB-B.ST", "SHB-A.ST", "SINCH.ST", "HM-B.ST", 
 "TELIA.ST", "SBB-B.ST", "SEB-A.ST", "ALFA.ST", 
 "ESSITY-B.ST", "NIBE-B.ST", "INVE-B.ST", "HEXA-B.ST", 
 "AZN.ST", "BOL.ST", "KINV-B.ST", "SCA-B.ST", 
 "ERIC-B.ST", "ATCO-A.ST", "ATCO-B.ST", "GETI-B.ST", "SAND.ST", 
 "ELUX-B.ST", "VOLV-B.ST"]

data = yf.download(tickers=tickers, start=start, end=end, group_by="ticker", auto_adjust=True)


df = data.stack(level=0).reset_index()
df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
df = df.set_index('Date')

start, end = df.index.min(), df.index.max()
ticker_dates = df.groupby('Ticker').apply(
    lambda x: pd.Series({
        'first': x.index.min(),
        'last' : x.index.max()
    })
)
full_tickers = ticker_dates[
    (ticker_dates['first'] <= start) &
    (ticker_dates['last']  >= end)
].index.tolist()

df = df[df['Ticker'].isin(full_tickers)]

shares_out = {}
for t in tickers:
    info = yf.Ticker(t).info
    shares_out[t] = info.get('sharesOutstanding', np.nan)

df['SharesOutstanding'] = df['Ticker'].map(shares_out)
df['MarketCap'] = df['Close'] * df['SharesOutstanding']

daily_mcap = df.groupby('Date')['MarketCap'].transform('sum')
df['Weight'] = df['MarketCap'] / daily_mcap

df['Return'] = df.groupby('Ticker')['Close'].pct_change() #.fillna(0)
df['LogReturn'] = np.log1p(df['Return']) #.fillna(0)


df['WeightedRet'] = df['Return'] * df['Weight']
df['WeightedRet'] = df['WeightedRet'] #.fillna(0)

daily_index_ret = df.groupby('Date')['WeightedRet'].sum()
index_value = (1 + daily_index_ret).cumprod()
index_value.name = 'MCAP_22_Index'

df['Volume'] = (
    df.groupby('Ticker')['Volume']
      .transform(lambda x: x.replace(0, np.nan).ffill().bfill())
)

df['SMA20'] = (
    df
    .groupby('Ticker')['Close']
    .transform(lambda x: ta.trend.SMAIndicator(close=x, window=20).sma_indicator())
    .fillna(0)
)

df['EMA20'] = (
    df
    .groupby('Ticker')['Close']
    .transform(lambda x: ta.trend.EMAIndicator(close=x, window=20).ema_indicator())
)

df['RSI14'] = (
    df
    .groupby('Ticker')['Close']
    .transform(lambda x: ta.momentum.RSIIndicator(close=x, window=14).rsi())
)

df['ReturnVola20'] = (
    df
    .groupby('Ticker')['Return']
    .transform(lambda x: x.rolling(window=20).std())
)

for col in ['SMA20','EMA20','RSI14','ReturnVola20']:
    df[col] = df.groupby('Ticker')[col] \
                 .transform(lambda x: x.fillna(method='bfill').fillna(method='ffill'))


df.to_csv("OMXS22_raw_features.csv")
index_value.to_frame().to_csv("MCAP_22_Index_raw.csv") 

cols_model = [
    'Ticker',        
    'Return',        
    'LogReturn',
    'Volume',        
    'SMA20',
    'EMA20',
    'RSI14',
    'ReturnVola20'
]

macro_tickers = ['^OMX', 'SEK=X', '^VIX', 'BZ=F']

macro_data = yf.download(tickers=macro_tickers, start=start, end=end, group_by="ticker", auto_adjust=True)

macro_df = pd.DataFrame({
    "OMXS30_ret":   macro_data["^OMX"]["Close"].pct_change(),
    "USD_SEK_ret":  macro_data["SEK=X"]["Close"].pct_change(),
    "VIX_ret":      macro_data["^VIX"]["Close"].pct_change(),
    "BRENT_ret":    macro_data["BZ=F"]["Close"].pct_change()
})

macro_df = macro_df.ffill().bfill()


df_model = df[cols_model].copy()
df_model = df_model.dropna(subset=cols_model)
df_model = (
    df_model.reset_index()
    .merge(macro_df, how='left', on='Date')
    .set_index('Date')
)

print(df_model.head())

df_model.to_csv("OMXS22_model_features_raw.csv")