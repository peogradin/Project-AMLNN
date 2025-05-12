#%%
# Plot predictions vs actual on test set
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(idx, tickers, loader, model, n_start=0, n_stop=20000):
    preds_idx = []
    trues_idx = []
    ticker = tickers[idx]
    with torch.no_grad():
        for Xb, yb in loader:
            preds = model(Xb)  # (B, A, H)

            if preds.dim() < 3:
                preds = preds.unsqueeze(-1)
            if yb.dim() < 3:
                yb = yb.unsqueeze(-1)
            
            preds_real = loader.dataset.inverse_transform(preds)
            yb_real = loader.dataset.inverse_transform(yb)
            all_preds = preds_real.cpu().numpy()
            all_trues = yb_real.cpu().numpy()
            preds_idx.append(preds_real[:, idx].cpu().numpy())
            trues_idx.append(yb_real[:, idx].cpu().numpy())

    preds_idx = np.concatenate(preds_idx)
    trues_idx = np.concatenate(trues_idx)
    preds_idx = preds_idx 
    trues_idx = trues_idx 
    
    plt.figure()
    plt.plot(trues_idx[n_start:n_stop], label='Actual')
    plt.plot(preds_idx[n_start:n_stop], label='Predicted')
    plt.xlabel('Sample index')
    plt.ylabel('Cumulative return for ' + ticker)
    plt.title('Predicted vs Actual on Training Set for ' + ticker)
    plt.legend()
    plt.show()

    # Print accuracy
    accuracy = (np.abs(all_preds - all_trues)/all_trues).mean()
    accuracy = np.mean(accuracy)
    print(f"Accuracy: {accuracy:.4f}")