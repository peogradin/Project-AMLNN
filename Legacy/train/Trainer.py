#%%
import torch
import numpy as np

# %%

class Trainer:
    """
    A generic trainer for PyTorch models that handles both:
        - models expecting a single tensor input (e.g. full-universe/multi-asset models)
        - models expecting a dict-input (e.g. ticker-embedded models)
    """

    def __init__(self, model, optimizer, criterion, device=None):
        """
        Args:
            model : nn.Module
            optimizer : torch.optim.Optimizer
            criterion : loss function, e.g. nn.MSELoss()
            device : 'cpu', 'cuda' or None to auto-detect
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def _parse_batch(self, batch):
        """
        Unpack a batch from DataLoader. Supports:
            - (x, y) where X is a tensor
            - ({...}, y) where the first element is a dict of tensors
        Returns (inputs, targets) on the correct device
        """
        inputs, targets = batch
        targets = targets.to(self.device)

        if isinstance(inputs, dict):
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            return inputs, targets
        else:
            return inputs.to(self.device), targets
        
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            inputs, targets = self._parse_batch(batch)
            self.optimizer.zero_grad()

            preds = self.model(inputs)
            loss = self.criterion(preds, targets)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                inputs, targets = self._parse_batch(batch)
                preds = self.model(inputs)
                total_loss += self.criterion(preds, targets).item()
        return total_loss / len(loader)
    
    def fit(self,
            train_loader,
            val_loader=None,
            epochs=50,
            patience=5,
            save_path="best_model.pt"):
        """
        Training loop with optional early stopping.
        Returns a dict: {'train_loss': [...], 'val_loss': [...]}
        """

        best_val = np.inf
        patience_ctr = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(1, epochs +1):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            if val_loader:
                val_loss = self.eval_epoch(val_loader)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch}/{epochs} - "
                      f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
                
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(self.model.state_dict(), save_path)
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr > patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            else:
                print(f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f}")

        return history