#%%
# Plot predictions vs actual on test set
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(idx, tickers, loader, model, n_start=0, n_stop=20000):
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
    ticker = tickers[idx]
    with torch.no_grad():
        for Xb, yb in loader:
            preds = model(Xb)  # (B, A, H)
            
            # B, A, H = preds.shape
            # B, T, A, F = Xb.shape
            # if preds.dim() < 3:
            #     preds = preds.view(B, A, H)
            # if yb.dim() < 3:
            #     yb = yb.view(B, A, H)
            # if Xb.dim() < 4:
            #     Xb = Xb.view(B, T, A, F) # (B, T, A, F)

            preds_real = loader.dataset.inverse_transform(preds)
            yb_real = loader.dataset.inverse_transform(yb)

            B, AH = preds_real.shape
            A = len(loader.dataset.tickers)
            H = len(loader.dataset.horizon)

            preds_real = preds_real.view(B, A, H)
            yb_real = yb_real.view(B, A, H)

            all_preds = preds_real.cpu().numpy()
            all_trues = yb_real.cpu().numpy()

            preds_idx.append(preds_real[:, idx, 5].cpu().numpy())
            trues_idx.append(yb_real[:, idx, 5].cpu().numpy())

    preds_idx = np.concatenate(preds_idx)
    trues_idx = np.concatenate(trues_idx)
    preds_idx = preds_idx 
    trues_idx = trues_idx 
    if loader.dataset.num_permutations <= 1:
        dates = loader.dataset.get_target_dates()
        dates = dates[n_start:n_stop]
        plt.figure()
        plt.plot(dates, trues_idx[n_start:n_stop], label='Actual')
        plt.plot(dates, preds_idx[n_start:n_stop], label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f"{ticker} - Predictions vs Actual with {loader.dataset.horizon[-1]} days horizon")
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
        plt.title(f"{ticker} - Predictions vs Actual with {loader.dataset.horizon[-1]} days horizon")
        plt.legend()
        plt.show()

    # Print accuracy
    accuracy = (np.abs(all_preds - all_trues)/all_trues).mean()
    accuracy = np.mean(accuracy)
    print(f"Accuracy: {accuracy:.4f}")
    
# %%
