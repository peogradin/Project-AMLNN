# Backtest
#%%
from portfolio_optimizers.portfolio_optimizer import PortfolioWeightOptimizer
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def backtest_multi_asset(ds, model, optimizer, df_prices, index_df=None, plot=True, trading_cost=0.0):
    """
    Backtest a multi-asset portfolio strategy using a trained model and optimizer.
    Args:
        ds (Dataset): Dataset containing the data for backtesting.
        model (nn.Module): Trained model for making predictions.
        optimizer (PortfolioWeightOptimizer): Portfolio weight optimizer.
        df_prices (pd.DataFrame): DataFrame containing historical prices of assets.
        index_df (pd.DataFrame, optional): DataFrame containing historical prices of a benchmark index.
        plot (bool, optional): Whether to plot the results. Defaults to True.
    Returns:
        dict: Dictionary containing portfolio values, dates, and weights (and index values, if any).
    """
    model.eval()
    device = next(model.parameters()).device
    tickers = ds.tickers
    A = len(tickers)
    T = ds.window
    F = len(ds.features)
    target_col = ds.target_col
    target_col_idx = ds.features.index(target_col)
    H = len(ds.horizon)
    max_h = max(ds.horizon)

    portfolio_vals = [1.0]
    portfolio_dates = [ds.sequence_dates[0]]
    portfolio_weights = []
    start_date = ds.sequence_dates[0]
    # Eventuellt: hämta index
    if index_df is not None:
        # index_df = index_df.set_index('Date').sort_index()
        index_vals = [index_df.loc[start_date, 'Close']]  # börja på samma dag som portföljen
    else:
        index_vals = None

    df_pivot = df_prices.pivot(columns='Ticker', values='Close')
    df_pivot = df_pivot.ffill().bfill()
    df_returns = df_pivot.pct_change().fillna(0)
    

    with torch.no_grad():
        for i in (range(len(ds))):
            X, y = ds[i]
            X = X.unsqueeze(0)

            preds = model(X)
            preds_real = ds.inverse_transform(preds, i)
            preds_real = preds_real.view(1, A, H)

            X_real = ds.inverse_feature_transform(X, i)
            X_real = X_real.view(1, T, A, F)
            X_tgt = X_real[:, :, :, target_col_idx]
            X_tgt = X_tgt.view(1, T, A, 1)

            weights = optimizer(X_tgt, preds_real)
            weights = weights[0].cpu().numpy()

            start_date = ds.sequence_dates[i]
            end_date = ds.target_dates[i]

            date_mask = (df_pivot.index > start_date) & (df_pivot.index <= end_date)
            daily_returns = df_returns.loc[date_mask]
            if daily_returns.empty:
                continue
            portfolio_returns = daily_returns.values @ weights
            previous_weights = portfolio_weights[-1] if portfolio_weights else np.zeros(A)
            buy_cost = np.sum(np.max(weights - previous_weights, 0)) * trading_cost
            portfolio_vals[-1] *= (1 - buy_cost)

            scaled_returns = portfolio_vals[-1] * np.cumprod(1 + portfolio_returns)
            print(f"Scaled returns: {scaled_returns}")
            print(f"Weights: {weights}")
            print(f"Full horizon predictions: {preds_real[0, :, -1]}")
            print(f"Last know prices: {X_real[0, -1, :, 0]}")
            print(f"first horizon target: {ds.inverse_transform(y.unsqueeze(0), i).view(1, A, H)[0, :, 0]}")
            print(f"Last known price date: {start_date}")
            print(f"Horizon date: {end_date}")
            max_weight_idx = np.argmax(weights)
            max_weight_ticker = tickers[max_weight_idx]
            print(f"Max weight ticker: {max_weight_ticker}, weight: ({weights[max_weight_idx]})")
            print(f"Chosen stock start price: {df_pivot.loc[start_date, max_weight_ticker]}")
            print(f"Chosen stock end price: {df_pivot.loc[end_date, max_weight_ticker]}")
            print(f"Chosen stock returns: {(df_pivot.loc[end_date, max_weight_ticker] - df_pivot.loc[start_date, max_weight_ticker]) / df_pivot.loc[start_date, max_weight_ticker]}")
            print(f"Chosen stock predicted returns: {preds_real[0, max_weight_idx, -1]}")
            print(f"Scaled returns over chosen period: {(scaled_returns[-1] - scaled_returns[0]) / scaled_returns[0]}")
            print("\n\n\n\n")

            portfolio_vals.extend(scaled_returns.tolist())
            portfolio_dates.extend(daily_returns.index.tolist())
            portfolio_weights.append(weights.tolist())
            
            if index_df is not None:
                index_segment = index_df.loc[daily_returns.index]['Close']
                #index_segment = index_segment / index_segment.iloc[0] * index_vals[-1]
                index_vals.extend(index_segment.tolist())
    result = {
        "dates": portfolio_dates,
        "values": portfolio_vals,
        "weights": portfolio_weights,
    }

    if index_vals is not None:
        result["benchmark"] = index_vals

    # === Optional plot ===
    if plot:
        print(f"Len(portfolio_dates): {len(portfolio_dates)}")
        print(f"Len(portfolio_vals): {len(portfolio_vals)}")
        print(f"Len(portfolio_weights): {len(portfolio_weights)}")
        print(f"Len(index_vals): {len(index_vals)}")
        print(f"dates sorted? {sorted(portfolio_dates) == portfolio_dates}")
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_dates, portfolio_vals[:], label="Strategy")
        if index_vals is not None:
            plt.plot(portfolio_dates, index_vals[:]/index_vals[0], label="Benchmark (Index)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio value")
        plt.title("Backtest: Multi-asset strategy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return result