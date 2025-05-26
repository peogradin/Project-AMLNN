import numpy as np
import matplotlib.pyplot as plt
def softmax_np(matrix):
    col_max = np.max(matrix, axis=0, keepdims=True)
    exp_max_reduced_values = np.exp(matrix - col_max)

    col_sums = np.sum(exp_max_reduced_values, axis=0, keepdims=True)
    softmaxed_value = exp_max_reduced_values/col_sums
    return softmaxed_value
def expand_weights(weights, horizon, num_of_test_points, only_use_positive_weights=False):
    if only_use_positive_weights:
        weights = np.clip(weights, a_min=0.0, a_max=None)
    expanded_weights = np.repeat(weights, horizon, axis=1)
    return expanded_weights[:,:num_of_test_points]

def allocate_from_weights(weights, horizon, num_of_test_points):
    all_weights = np.array(weights)
    all_weights_expanded = expand_weights(all_weights, horizon, num_of_test_points, only_use_positive_weights=True)

    allocations = softmax_np(all_weights_expanded)
  
    return allocations

def plot_allocations(alloc, tickers):
    n_assets, total_steps = alloc.shape
    print(total_steps)
    
    

    # Time axis
    t = np.arange(total_steps)

    # Compute cumulative allocations for stacking
    cumsum = np.cumsum(alloc, axis=0)

    plt.figure()
    # Plot filled areas for each asset
    for i in range(n_assets):
        bottom = cumsum[i-1] if i > 0 else np.zeros_like(t)
        top = cumsum[i]
        plt.fill_between(t, bottom, top, step='post', label=tickers[i])
    
    plt.xlabel('Time Step')
    plt.ylabel('Allocation')
    plt.title('Held Asset Allocations Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()

def retrieve_test_closing_price(df, chosen_ticker, num_test_points):
    df = df[df["Ticker"] == chosen_ticker]
    df = df.iloc[20:]

    prices = df["Close"].values  
    test_prices = prices[-num_test_points-1:]
    return test_prices  

def calculate_asset_returns(df, tickers, horizon, num_of_test_points):
    #print(num_of_test_points)
    returns = np.zeros((len(tickers), num_of_test_points))
    for i, ticker in enumerate(tickers):
        closing_prices = retrieve_test_closing_price(df, ticker, num_of_test_points)
        #print(closing_prices)
        for t in range(returns.shape[1]):
            if t*horizon + horizon < len(closing_prices):
                real_future_price = closing_prices[t*horizon+horizon]
                #print('future price: ', real_future_price)
                
                real_last_price = closing_prices[t*horizon]
                #print('last price: ', real_last_price)
                return_t = (real_future_price - real_last_price)/real_last_price
                returns[i, t*horizon] = return_t
    
    expanded_returns = returns# expand_weights(returns, horizon, num_of_test_points, only_use_positive_weights=False)

    return expanded_returns

def calculate_asset_returns_cont(df, tickers, num_of_test_points):
    #print(num_of_test_points)
    returns = np.zeros((len(tickers), num_of_test_points))
    for i, ticker in enumerate(tickers):
        closing_prices = retrieve_test_closing_price(df, ticker, num_of_test_points)
        #print(closing_prices)
        #for t in range(num_of_test_points):
        #old_price = closing_prices[t]
        #new_price = closing_prices[t+1]
        current_ret = np.diff(closing_prices, axis = 0) / closing_prices[:-1]
       
        
        returns[i, :] = current_ret
                 
    return returns

def calculate_portfolio_returns(allocations, returns):
    return np.sum(allocations * returns, axis=0)

def calculate_cumulative_portfolio_returns(portfolio_returns, start_value=1):
    cum_portfolio_returns = start_value*np.cumprod(1 + portfolio_returns)
    return cum_portfolio_returns
