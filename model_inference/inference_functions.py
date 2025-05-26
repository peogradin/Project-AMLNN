import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_predictions(model, window_size, real_data, num_of_future_predictions, horizon):
    model.eval()

    train_mean = real_data.mean(axis=0)
    train_std = real_data.std(axis=0)
    train_mean_torch = torch.tensor(train_mean, dtype=torch.float32)
    train_std_torch = torch.tensor(train_std, dtype=torch.float32)
    #print(train_mean_torch)
    #print(train_std_torch)


    last_window = torch.tensor(real_data[-window_size-num_of_future_predictions:-num_of_future_predictions], dtype=torch.float32)
    last_window = (last_window - train_mean_torch ) / train_std_torch
    print('last window shape: ',last_window.shape)
    #print(last_window)

    test_real_data_torch = torch.tensor(real_data[-num_of_future_predictions:], dtype=torch.float32)


    test_real_data_torch = (test_real_data_torch - train_mean_torch)/ train_std_torch
    print('test data shape', test_real_data_torch.shape)

    #print(test_real_data_torch)

    preds_norm = np.zeros(num_of_future_predictions)

    with torch.no_grad():
        for t in range(int(num_of_future_predictions/horizon)):
            x_in  = last_window.unsqueeze(0)
            pred_torch = model(x_in)
            #print('pred:', pred_torch)
            pred_norm = pred_torch.squeeze().item()

            preds_norm[t*horizon] = pred_norm
            #print('real: ', test_real_data_torch[t, 0])

            true_indicators = test_real_data_torch[t+horizon-1, :]
            true_indicators[0] = pred_norm

            next_data_point = torch.tensor(true_indicators, dtype=torch.float32)
            if horizon > 1:
                middle_points = torch.tensor(test_real_data_torch[(t)*horizon:(t+1)*horizon-1,:], dtype = torch.float32)
                #print(middle_points.shape)
                #print(next_data_point.unsqueeze(0).shape)
                #print('last window in: ',last_window.shape)
                modified_next_data_point = torch.cat([middle_points, next_data_point.unsqueeze(0)], dim=0)
            else:
                modified_next_data_point = next_data_point.unsqueeze(0)
            #print('mod shape', modified_next_data_point.shape)
            last_window = torch.cat([last_window[horizon:], modified_next_data_point], dim=0)
            #print('last window shape: ', last_window.shape)

    return preds_norm


def plot_pred_vs_real(preds_norm, real_data,target_idx, horizon, minimal_plotting = False):
    train_means = real_data.mean(axis=0)
    train_stds = real_data.std(axis=0)
    plt.plot(preds_norm, label='predictions')
    plt.plot((real_data[-200:][:,0] - train_means[target_idx])/train_stds[target_idx], label='real data')
    plt.legend()
    plt.title(f'Normalized predictions vs real data. Horizon = {horizon}')
    plt.show()

    if not minimal_plotting:
        real_val_pred = np.array(preds_norm) * train_stds[target_idx] + train_means[target_idx]
        real_val_true = np.array(real_data[-200:][:,0]) 

        plt.plot(real_val_pred, label='predictions')
        plt.plot(real_val_true, label='real data')
        plt.legend()
        plt.title('Real valued predictions vs real data')
        plt.show()

        old_data = real_data[-400:-200]
        real_old_data = np.array(old_data[:,0])

        real_long_pred = np.concat([real_old_data, real_val_pred])
        real_long_true = np.concat([real_old_data, real_val_true])

        plt.plot(real_long_pred, label='prediction')
        #plt.plot(real_long_true, label='treu values')
        plt.plot(real_data[-400:][:,0], label='real data')
        plt.legend()
        plt.title('Real valued predictions vs true values')
        plt.show()


def generate_predictions_and_returns(model, window_size, real_data, target_idx, num_of_future_predictions, horizon, kelly_clip = 1, print_values_for_allocation=False):
    model.eval()

    # 1) Compute in‐sample return variance (on training history)
    #    use price in column 0
    prices = real_data[:-num_of_future_predictions, 0]
    ret = (prices[horizon:] - prices[:-horizon]) / prices[:-horizon]
    sigma2 = np.var(ret)
    print('variance of returns for old data: ', sigma2)

    train_mean = real_data.mean(axis=0)
    train_std = real_data.std(axis=0)
    train_mean_torch = torch.tensor(train_mean, dtype=torch.float32)
    train_std_torch = torch.tensor(train_std, dtype=torch.float32)
    #print(train_mean_torch)
    #print(train_std_torch)


    last_window = torch.tensor(real_data[-window_size-num_of_future_predictions:-num_of_future_predictions], dtype=torch.float32)
    last_window = (last_window - train_mean_torch ) / train_std_torch
    print('start window shape: ',last_window.shape)
    #print(last_window)

    test_real_data_torch = torch.tensor(real_data[-num_of_future_predictions:], dtype=torch.float32)


    test_real_data_torch_norm = (test_real_data_torch - train_mean_torch)/ train_std_torch
    print('test data shape', test_real_data_torch.shape)

    #print(test_real_data_torch)

    preds_norm = np.zeros(num_of_future_predictions)
    weights = np.zeros(num_of_future_predictions)
    pnl = np.zeros(num_of_future_predictions)

    with torch.no_grad():
        for t in range(int(num_of_future_predictions/horizon)-1):

            idx_start = t * horizon
            idx_end = idx_start + horizon

            x_in  = last_window.unsqueeze(0)
            pred_torch = model(x_in)
            #print('pred:', pred_torch)
            pred_norm = pred_torch.squeeze().item()

            preds_norm[idx_start] = pred_norm
            #print('real: ', test_real_data_torch[t, 0])

            last_price_real = test_real_data_torch[idx_start, 0] if t>0 else real_data[-num_of_future_predictions-1, 0]
            #print(last_price_real)
            pred_price = pred_norm * train_std_torch[target_idx] + train_mean_torch[target_idx]
            pred_ret = (pred_price - last_price_real)/last_price_real
            #print(pred_ret.shape)

            #kelly weight
            k = pred_ret / np.sqrt(sigma2) if sigma2 > 1e-8 else 0.0
            k = np.clip(k, -kelly_clip, kelly_clip)
            #print(k.shape)

            weights[idx_start] = k
            
            true_horizon_price = test_real_data_torch[idx_end, 0]
            if print_values_for_allocation:
                print('x_in shape: ', x_in.shape)
                print('windows last price: ',x_in[0,-1,0] *train_std_torch[target_idx] + train_mean_torch[target_idx])
                print(f'period from idx: ', idx_start, ' to idx ', idx_end)
                
                print(f'prediction at {horizon}: ', pred_price)
                print('last price from test data: ', last_price_real)
                print(f'real price {horizon} steps forward: ', true_horizon_price)
                print(f'kelly weight: ', {k})
                print(' ')
            realized_ret = (true_horizon_price - last_price_real) / last_price_real
            pnl[idx_start] = k * realized_ret

            true_row = test_real_data_torch_norm[idx_end-1, :]# if t>0 else real_data[-num_of_future_predictions-1, 0]
            #true_indicators[0] = pred_norm

            next_data_point = torch.tensor(true_row, dtype=torch.float32)
            if horizon > 1:
                middle_points = torch.tensor(test_real_data_torch_norm[idx_start:idx_end-1,:], dtype = torch.float32)
                #print(middle_points.shape)
                #print(next_data_point.unsqueeze(0).shape)
                #print('last window in: ',last_window.shape)
                modified_next_data_point = torch.cat([middle_points, next_data_point.unsqueeze(0)], dim=0)
            else:
                modified_next_data_point = next_data_point.unsqueeze(0)
            #print('mod shape', modified_next_data_point.shape)
            last_window = torch.cat([last_window[horizon:], modified_next_data_point], dim=0)
            #print('last window shape: ', last_window.shape)

    return preds_norm, weights, pnl

def generate_weights_from_predictions(model, window_size, real_data, target_idx, num_of_future_predictions, horizon, kelly_clip = 1, print_values_for_allocation=False):
    model.eval()

    # 1) Compute in‐sample return variance (on training history)
    #    use price in column 0
    prices = real_data[:-num_of_future_predictions, 0]
    ret = (prices[horizon:] - prices[:-horizon]) / prices[:-horizon]
    sigma2 = np.var(ret)
    print('variance of returns for old data: ', sigma2)

    train_mean = real_data.mean(axis=0)
    train_std = real_data.std(axis=0)
    train_mean_torch = torch.tensor(train_mean, dtype=torch.float32)
    train_std_torch = torch.tensor(train_std, dtype=torch.float32)
    #print(train_mean_torch)
    #print(train_std_torch)


    last_window = torch.tensor(real_data[-window_size-num_of_future_predictions:-num_of_future_predictions], dtype=torch.float32)
    last_window = (last_window - train_mean_torch ) / train_std_torch
    print('start window shape: ',last_window.shape)
    #print(last_window)

    test_real_data_torch = torch.tensor(real_data[-num_of_future_predictions:], dtype=torch.float32)


    test_real_data_torch_norm = (test_real_data_torch - train_mean_torch)/ train_std_torch
    print('test data shape', test_real_data_torch.shape)

    #print(test_real_data_torch)

    preds_norm = np.zeros(num_of_future_predictions)
    weights = np.zeros(int(num_of_future_predictions/horizon))
    

    with torch.no_grad():
        for t in range(int(num_of_future_predictions/horizon)-1):

            idx_start = t * horizon
            idx_end = idx_start + horizon

            x_in  = last_window.unsqueeze(0)
            pred_torch = model(x_in)
            #print('pred:', pred_torch)
            pred_norm = pred_torch.squeeze().item()

            preds_norm[idx_start] = pred_norm
            #print('real: ', test_real_data_torch[t, 0])

            last_price_real = test_real_data_torch[idx_start, 0] if t>0 else real_data[-num_of_future_predictions-1, 0]
            #print(last_price_real)
            pred_price = pred_norm * train_std_torch[target_idx] + train_mean_torch[target_idx]
            pred_ret = (pred_price - last_price_real)/last_price_real
            #print(pred_ret.shape)

            #kelly weight
            k = pred_ret / np.sqrt(sigma2) if sigma2 > 1e-8 else 0.0
            k = np.clip(k, -kelly_clip, kelly_clip)
            #print(k.shape)

            weights[t] = k
            
            true_horizon_price = test_real_data_torch[idx_end, 0]
            if print_values_for_allocation:
                print('x_in shape: ', x_in.shape)
                print('windows last price: ',x_in[0,-1,0] *train_std_torch[target_idx] + train_mean_torch[target_idx])
                print('windows last2 price: ',x_in[0,-2,0] *train_std_torch[target_idx] + train_mean_torch[target_idx])
                print('windows last3 price: ',x_in[0,-3,0] *train_std_torch[target_idx] + train_mean_torch[target_idx])

                print(f'period from idx: ', idx_start, ' to idx ', idx_end)
                
                print(f'prediction at {horizon}: ', pred_price)
                print('last price from test data: ', last_price_real)
                print(f'real price {horizon} steps forward: ', true_horizon_price)
                print(f'kelly weight: ', {k})
                print(' ')
            
            

            true_row = test_real_data_torch_norm[idx_end, :]# if t>0 else real_data[-num_of_future_predictions-1, 0]
            #true_indicators[0] = pred_norm

            next_data_point = torch.tensor(true_row, dtype=torch.float32)
            if horizon > 1:
                middle_points = torch.tensor(test_real_data_torch_norm[idx_start+1:idx_end,:], dtype = torch.float32)
                #print(middle_points.shape)
                #print(next_data_point.unsqueeze(0).shape)
                #print('last window in: ',last_window.shape)
                modified_next_data_point = torch.cat([middle_points, next_data_point.unsqueeze(0)], dim=0)
            else:
                modified_next_data_point = next_data_point.unsqueeze(0)
            #print('mod shape', modified_next_data_point.shape)
            last_window = torch.cat([last_window[horizon:], modified_next_data_point], dim=0)
            #print('last window shape: ', last_window.shape)

    return preds_norm, weights

def plot_realized_returns(pnl, plot_only_cumulative = False):
    if not plot_only_cumulative:
        plt.figure()
        plt.plot(pnl, marker='o')
        plt.title('Daily P&L from Kelly Strategy')
        plt.xlabel('Step')
        plt.ylabel('Daily Return')
        plt.tight_layout()
        plt.show()

    # 2) Plot cumulative P&L
    cumulative_pnl = np.cumsum(pnl)
    plt.figure()
    plt.plot(cumulative_pnl, marker='o')
    plt.title('Cumulative P&L from Kelly Strategy')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Return')
    plt.tight_layout()
    plt.show()

def test_model_output_percentage_diff(model, data_loader, tm, tstd, target_idx, num_tests=20):
    
    model.eval()

    all_preds, all_true = [], []
    first_batch = None  # we’ll keep this for the per-sample print-out

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            #x_batch = x_batch.to(device)
            #y_batch = y_batch.to(device)

            # Forward pass
            out = model(x_batch)
            if isinstance(out, (tuple, list)):      # ignore extra returns (e.g. hidden state)
                out = out[0]

            all_preds.append(out.squeeze(-1)* tstd[target_idx] + tm[target_idx])        # ensure shape (batch,)
            all_true.append(y_batch.squeeze(-1) * tstd[target_idx] + tm[target_idx])

            # Save the very first batch for pretty printing later
            if first_batch is None:
                first_batch = (x_batch.cpu(), y_batch.cpu(), out.cpu())

    # ------------------------------------------------------------------
    # Stack everything and compute Mean Absolute Percentage Error (MAPE)
    # ------------------------------------------------------------------
    preds  = torch.cat(all_preds)          # (N,)
    truth  = torch.cat(all_true)           # (N,)

    abs_diff = (preds - truth).abs()
    percentage_diff = abs_diff / truth.abs().clamp(min=1e-8) * 100  # % error
    mean_percent_error = percentage_diff.mean().item()

    print(f"\nMean absolute percentage error (MAPE): {mean_percent_error:.2f}%")

    # ------------------------------------------------------------------
    # Show a few individual samples (from the first batch)
    # ------------------------------------------------------------------
    x_b, y_b, p_b = first_batch
    num_tests = min(num_tests, x_b.size(0))

    print(f"\nFirst-batch sample comparisons (n = {num_tests}):")
    for i in range(num_tests):
        # Example feature: last temperature in feature-index 1
        last_input_temp = (
            x_b[i, -1, 1].item()
            if x_b.dim() == 3 and x_b.size(2) > 1
            else float('nan')
        )
        pct = percentage_diff[i].item()
        print(
            f"  Sample {i+1:2d}: "
            f"Last input temp = {last_input_temp:6.2f}, "
            f"True = {y_b[i].item()* tstd[target_idx] + tm[target_idx]:6.2f}, "
            f"Pred = {p_b[i].item()* tstd[target_idx] + tm[target_idx]:6.2f}, "
            f"Pct diff = {pct:6.2f}%"
        )
    return mean_percent_error