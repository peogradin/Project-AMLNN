#%%
import yfinance as yf

data = yf.download(["AAPL", "^OMX"], start="2010-01-01", end="2025-04-29")
# %%
print(data.head())

import pandas as pd
# convert to pandas dataframe
data = pd.DataFrame(data)
# save to csv
data.to_csv("AAPL_OMX.csv")
#%%
print(data)
import matplotlib.pyplot as plt
# plot the data
data['Close'].plot(title="AAPL and OMX Close Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(["AAPL", "OMX"])
plt.show()
# %%
