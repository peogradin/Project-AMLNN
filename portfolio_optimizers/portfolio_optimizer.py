#%%
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

class PortfolioWeightOptimizer:
    """
    Portfolio weight optimizer for different strategies.
    Args:
        strategy (str): Optimization strategy. Options: "softmax_pred", "sharpe_opt", "mean_variance_opt", "equal_weights"
        multi_horizon (bool): Whether to use multi-horizon predictions
        risk_aversion (float): Risk aversion parameter for mean-variance optimization
    """

    def __init__(self, strategy="softmax_pred", multi_horizon=False, risk_aversion=1.0):
        assert strategy in ["softmax_pred", "sharpe_opt", "mean_variance_opt", "equal_weights"], f"Invalid strategy {strategy}"
        self.strategy = strategy
        self.multi_horizon = multi_horizon
        self.risk_aversion = risk_aversion

    def __call__(self, X, preds):
        """
        Args:
            X (torch.Tensor): Input data (B, T, A, 1). Must have target column at last dimension. Must be real asset values.
            preds (torch.Tensor): Predictions (B, A, H)
        """
        assert X.dim() == 4, f"Expected 4D tensor for X, got {X.dim()}D"
        assert preds.dim() == 3, f"Expected 3D tensor for preds, got {preds.dim()}D"
        

        if self.strategy == "sharpe_opt":
            return self._optimize_sharpe_batch(X, preds)

        elif self.strategy == "softmax_pred":
            return F.softmax(preds[:, :, -1], dim=1)
        
        elif self.strategy == "mean_variance_opt":
            return self._optimize_mean_variance_batch(X, preds)
        
        elif self.strategy == "equal_weights":
            A = preds.shape[1]
            return torch.full((preds.shape[0], A), 1.0 / A, device=preds.device)
    
    def _optimize_sharpe_batch(self, X, preds):
        if not self.multi_horizon:
            raise ValueError("Must have a multi-horizon (i.e. sequence predictions) for sharpe_opt strategy")
        
        B, A, H = preds.shape
        weights = []
        
        for i in range(B):
            r = preds[i].detach().cpu().numpy()
            #r0 = r[:, [0]]
            last_price = X[i, -1, :, 0].detach().cpu().numpy()
            r = r / last_price[:, None] - 1.0
            #print(f"Last price: {last_price} \n r[:, [0]]: {r0}")
            past_prices = X[i, :, :, 0].detach().cpu().numpy() # (W, A)
            past_returns = past_prices / past_prices[0, :] - 1.0
            cov = np.cov(past_returns.T)

            def negative_log_sharpe(w):
                """
                w: (A,), r: (A, H)
                returns: negative log sharpe (H,)
                """
                #print(f"w: {w.shape}", f"r: {r.shape}")
                portfolio_return = w @ r
                portfolio_risk = np.std(w @ past_returns.T)
                
                #print(f"Portfolio return: {portfolio_return.shape}")
                # print(f"Portfolio return: {portfolio_return}")
                
                log_returns = np.log(1 + portfolio_return)
                log_risk = np.log(1 + w @ past_returns.T)
                # print(f"Log returns: {log_returns}")
                ln_E_R = np.mean(log_returns)
                ln_std_R = np.std(log_risk)
                #print(f"Ln_E_R + Ln_std_R: -{ln_E_R} + {ln_std_R}")
                return -self.risk_aversion*ln_E_R + ln_std_R
                #return - self.risk_aversion * np.mean(portfolio_return) / portfolio_risk
            
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0.0, 1.0)] * A
            init_w = np.full(A, 1.0 / A)

            result = minimize(negative_log_sharpe, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                w_opt = result.x
            else:
                print(f"[Warning] Optimization failed at sample {i}. Using equal weights.")
                w_opt = init_w
            weights.append(torch.tensor(w_opt, dtype=torch.float32).to(preds.device))
        
        return torch.stack(weights).to(preds.device) # (B, A)
    
    def _optimize_mean_variance_batch(self, X, preds):
        if not self.multi_horizon:
            raise ValueError("Must have a multi-horizon (i.e. sequence predictions) for mean_variance_opt strategy")
        
        B, A, H = preds.shape
        weights = []

        for i in range(B):
            r = preds[i].detach().cpu().numpy()
            last_price = X[i, -1, :, 0].detach().cpu().numpy()
            r = r / last_price[:, None] - 1.0
            past_prices = X[i, :, :, 0].detach().cpu().numpy()
            past_returns = past_prices / past_prices[0, :] - 1.0

            mu = r.mean(axis=1)
            cov = np.cov(r)


            def objective(w):
                #print("first", w @ cov @ w, "second", mu @ w)
                portfolio_return = w @ r
                mean_return = np.mean(portfolio_return)
                std_return = np.std(w @ past_returns.T)
                #return w @ cov @ w - self.risk_aversion * (mu @ w) 
                return std_return - self.risk_aversion * mean_return

            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0.0, 1.0)] * A
            init_w = np.full(A, 1.0 / A)

            result = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                w_opt = result.x
            else:
                print(f"[Warning] Optimization failed at sample {i}. Using equal weights.")
                w_opt = init_w
            
            weights.append(torch.tensor(w_opt, dtype=torch.float32).to(preds.device))

        return torch.stack(weights).to(preds.device) # (B, A)
    
#%%
import numpy as np
if __name__ == "__main__":
    B = 1
    A = 22
    H = 10
    
    X = torch.rand(B, 100, A, 10)*500 + 1
    preds = torch.rand(B, A, H)*500 + 1
    optimizer = PortfolioWeightOptimizer(strategy="sharpe_opt", multi_horizon=True)
    weights = optimizer(X, preds)
    print(weights)
    print(weights.shape)

    optimizer = PortfolioWeightOptimizer(strategy="softmax_pred", multi_horizon=True)
    weights = optimizer(X, preds)
    print(weights)
    print(weights.shape)

    optimizer = PortfolioWeightOptimizer(strategy="mean_variance_opt", multi_horizon=True)
    weights = optimizer(X, preds)
    print(weights)
    print(weights.shape)

    optimizer = PortfolioWeightOptimizer(strategy="equal_weights", multi_horizon=True)
    weights = optimizer(X, preds)
    print(weights)
    print(weights.shape)
    
# %%
