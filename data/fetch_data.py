#%%

import yfinance as yf

tickers = ["SKF-B.ST", "NDA-SE.ST", "ASSA-B.ST",
 "ABB.ST", "SWED-A.ST", "TEL2-B.ST", "EVO.ST",
 "SAAB-B.ST", "SHB-A.ST", "SINCH.ST", "HM-B.ST", 
 "TELIA.ST", "SBB-B.ST", "SEB-A.ST", "ALFA.ST", 
 "ESSITY-B.ST", "NIBE-B.ST", "INVE-B.ST", "HEXA-B.ST", 
 "AZN.ST", "BOL.ST", "KINV-B.ST", "SCA-B.ST", 
 "ERIC-B.ST", "ATCO-A.ST", "GETI-B.ST", "SAND.ST", 
 "ELUX-B.ST", "VOLV-B.ST"]

data = yf.download(tickers=tickers, start="2000-01-01", end="2025-04-29")

print(data["Close"].head())

import pandas as pd

data = pd.DataFrame(data)

data.to_csv("OMXS30_underlying_raw.csv")
