import pytorch_lightning as pl
import torch
import torch.nn as nn

class TSTrainer(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        optimizer_cls,
        optimizer_kwargs: dict,
        scheduler_cls=None,
        scheduler_kwargs: dict = None,
    ):
        """
        A LightningModule that will train *any* model.

        Args:
          model          – your nn.Module
          loss_fn        – e.g. nn.MSELoss()
          optimizer_cls  – e.g. torch.optim.Adam
          optimizer_kwargs – kwargs for optimizer (lr, weight_decay…)
          scheduler_cls  – optional LR scheduler class
          scheduler_kwargs – kwargs for scheduler
        """
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        # logs to TensorBoard/console
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val/loss", val_loss, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        opt = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
        if self.scheduler_cls:
            sch = self.scheduler_cls(opt, **self.scheduler_kwargs)
            return [opt], [sch]
        return opt
