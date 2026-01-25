import torch
import lightning as pl

class BaseModel(pl.LightningModule):
    def __init__(self, criterion, optimizer_params, scheduler_params):
        super().__init__()
        self.criterion = criterion
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params


    def prepare_batch(self, batch):
        raise NotImplementedError

    def predict_batch(self, batch):
        raise NotImplementedError

    def configure_optimizers(self):
        if self.optimizer_params is None:
            self.optimizer_params = dict(name='Adam', lr=1e-4, weight_decay=1e-3)
        try:
            optimizer_class = getattr(torch.optim, self.optimizer_params.pop('name'))
        except:
            optimizer_class =torch.optim.Adam
        optimizer = optimizer_class(filter(lambda p: p.requires_grad, self.parameters()), **self.optimizer_params)

        if self.scheduler_params is not None:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler_params.pop('name'))
            monitor = self.scheduler_params.pop('monitor')
            self.reduce_lr_scheduler = scheduler_class(optimizer, **self.scheduler_params)
            scheduler = dict(scheduler=self.reduce_lr_scheduler, monitor=monitor)
            return dict(optimizer=optimizer, lr_scheduler=scheduler)
        return optimizer

    def training_step(self, batch, batch_idx):
        scores, labels = self.predict_batch(batch)
        loss = self.criterion(scores, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, labels = self.predict_batch(batch)
        loss = self.criterion(preds, labels)
        self.log("val_loss", loss, batch_size=labels.shape[0], on_epoch=True, on_step=False)
        return loss

