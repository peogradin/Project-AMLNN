#%%
# Plot predictions vs actual on test set
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(ticker_idx, tickers, dataset, model, n_start=0, n_stop=20000):
    """
    Plot predictions vs actual on test set, taking the last time step of the predicted sequence.
    Args:
        idx (int): Index of the ticker to plot
        tickers (list): List of tickers
        loader (DataLoader): DataLoader for the test set
        model (nn.Module): Trained model
        n_start (int): Start index for plotting
        n_stop (int): Stop index for plotting
    """

    preds_idx = []
    trues_idx = []
    all_preds = []
    all_trues = []
    ticker = tickers[ticker_idx]
    with torch.no_grad():
        for idx, (Xb, yb) in enumerate(dataset):
            preds = model(Xb.unsqueeze(0))  # (B, A*H)

            preds_real = dataset.inverse_transform(preds, idx)
            yb_real = dataset.inverse_transform(yb.unsqueeze(0), idx)

            B, AH = preds_real.shape
            A = len(dataset.tickers)
            H = len(dataset.horizon)

            preds_real = preds_real.view(B, A, H)
            yb_real = yb_real.view(B, A, H)

            all_preds.append(preds_real.cpu().numpy())
            all_trues.append(yb_real.cpu().numpy())

            preds_idx.append(preds_real[:, ticker_idx, -1].cpu().numpy())
            trues_idx.append(yb_real[:, ticker_idx, -1].cpu().numpy())

    preds_idx = np.concatenate(preds_idx)
    trues_idx = np.concatenate(trues_idx)
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    if dataset.num_permutations <= 1:
        dates = dataset.get_target_dates()
        dates = dates[n_start:n_stop]
        plt.figure()
        plt.plot(dates, trues_idx[n_start:n_stop], label='Actual')
        plt.plot(dates, preds_idx[n_start:n_stop], label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f"{ticker} - Predictions vs Actual with {dataset.horizon[-1]} days horizon")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        plt.figure()
        plt.plot(trues_idx[n_start:n_stop], label='Actual')
        plt.plot(preds_idx[n_start:n_stop], label='Predicted')
        plt.xlabel('Sample index')
        plt.ylabel('Price')
        plt.title(f"{ticker} - Predictions vs Actual with {dataset.horizon[-1]} days horizon")
        plt.legend()
        plt.show()

    # Print accuracy
    accuracy = (np.abs(all_preds - all_trues)/all_trues).mean()
    accuracy = np.mean(accuracy)
    print(f"Accuracy: {accuracy:.4f}")

    last_preds = all_preds[:, :, -1]  # shape (N, A)
    last_trues = all_trues[:, :, -1]
    accuracy_last_horizon = (np.abs(last_preds - last_trues) / last_trues).mean()
    print(f"Total accuracy on last horizon: {accuracy_last_horizon:.4f}")
    for i, ticker in enumerate(tickers):
        acc_i = (np.abs(last_preds[:, i] - last_trues[:, i]) / last_trues[:, i]).mean()
        print(f"{ticker}: {acc_i:.4f}")

    stock_accuracy = (np.abs(preds_idx - trues_idx)/trues_idx).mean()
    print(f"Stock accuracy: {stock_accuracy:.4f}")
    
# %%
