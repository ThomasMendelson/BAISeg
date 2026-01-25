import os
import sys
import yaml
import wandb
import random
import argparse
import torch
import torch.nn as nn

from easydict import EasyDict
import lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from dataset import CELL2DDataModule, CELL3DDataModule



from utils import FocalBCEWithLogits
from models.unet import Unet2dWrap, Unet2_5dWrap, Unet3dWrap


torch.cuda.empty_cache()


CUSTOM_LOSSES = {
    "FocalBCEWithLogits": FocalBCEWithLogits,
}
class ModelCheckpointSaveLast(Callback):
    def __init__(self, filename, dirpath):
        self.filename = filename
        self.dirpath = dirpath

    def on_train_end(self, trainer, pl_module):
        # if (trainer.current_epoch + 1) == trainer.max_epochs:
        base_name = self.filename.replace("{epoch:03d}", "epoch={epoch:03d}")
        base_name = base_name.format(epoch=trainer.current_epoch)
        ckpt_name = f"{base_name}.ckpt"
        os.makedirs(self.dirpath, exist_ok=True)
        path = os.path.join(self.dirpath, ckpt_name)

        version = 1
        while os.path.exists(path):
            ckpt_name = f"{base_name}-v{version}.ckpt"
            path = os.path.join(self.dirpath, ckpt_name)
            version += 1

        trainer.save_checkpoint(path)

def yaml_read(path):
    with open(os.path.realpath(path), "r") as fp:
        obj = yaml.safe_load(fp)
    return EasyDict(obj)


def init_logger(cfg, logger=None):
    # if not args.debug and not args.test and cfg.wandb:
    if not args.debug and cfg.wandb and not cfg.data.sys == 'CTC':
        logger = WandbLogger(project='unet', group=cfg.name, name=cfg.name, job_type='train',
                             save_dir=os.environ['HOME'], log_model=False, tags=cfg.wandb_tags)
        try:
            # Check if the experiment config has an update method
            if hasattr(logger.experiment.config, 'update'):
                print("in if hasattr(logger.experiment.config, 'update')")
                logger.experiment.config.update(cfg)
            else:
                print("Logger config does not have an update method.")
        except wandb.errors.Error as e:
            print("W&B bug: ", e)
    return logger


def get_criterion(cfg):
    criterion_list = []
    for lc in cfg:
        loss_cfg = dict(lc)
        name = loss_cfg.pop('name')
        criterion_class = getattr(nn, name, None)
        if criterion_class is None:
            print(f"name: {name}")
            print(f"CUSTOM_LOSSES.get(name): {CUSTOM_LOSSES.get(name)}")
            criterion_class = CUSTOM_LOSSES.get(name)
        if criterion_class is None:
            raise ValueError(f"Unknown loss name '{name}'. "
                             f"Must be torch.nn.* or one of {list(CUSTOM_LOSSES.keys())}.")
        if 'pos_weight' in loss_cfg:
            loss_cfg['pos_weight'] = torch.tensor(loss_cfg['pos_weight'], dtype=torch.float32)
        if 'delta' in loss_cfg:
            loss_cfg['delta'] = torch.tensor(loss_cfg['delta'], dtype=torch.float32)
        criterion = criterion_class(**loss_cfg) if len(loss_cfg) else criterion_class()
        criterion_list.append(criterion)
    return criterion_list


if __name__ == '__main__':
    # DEBUG = True
    PATH = "/home/thomasm/workspace/cells_lightning/cells/unet2d.yml"
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str, help="path to yaml config file", default=PATH)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-t", "--test", action="store_true", default=False)
    parser.add_argument("-i", "--input_dir", type=str)
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("-w", "--weights", type=str)
    args = parser.parse_args()
    assert os.path.exists(args.cfg), f"cfg file {args.cfg} does not exist!"
    cfg = yaml_read(args.cfg)

    logger = init_logger(cfg)


    gpu = cfg.trainer.pop('gpu')
    if cfg.data.sys == 'runai':
        ckpt_path = cfg.ckpt_path_runai + "unet/"
        devices = 'auto'
    elif cfg.data.sys == 'CTC':
        ckpt_path = ""
        cfg.model_to_load = args.weights
        cfg.data.test.img = [args.input_dir]
        cfg.model.output_dir = args.output_dir
    else:
        ckpt_path = cfg.ckpt_path_dgx + "unet/"
        if isinstance(gpu, list):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in gpu])
            print(f"os.environ['CUDA_VISIBLE_DEVICES']: {os.environ['CUDA_VISIBLE_DEVICES']}")
            # strategy = 'ddp_find_unused_parameters_false'
            devices = len(gpu)
        else:
            # strategy = None
            devices = [gpu]



    criterion = get_criterion(cfg.loss)

    # pre_trained_path = base_ckpt_path + "dino/" + cfg.model.pop(
    #     'pre_trained_path') if 'pre_trained_path' in cfg.model else None
    pre_trained_path = cfg.model.pop('pre_trained_path', None)
    if cfg._2d:
        data_module = CELL2DDataModule(cfg.data)

        if args.test:
            print(f"model_to_load: {cfg.model_to_load}")
            model = Unet2dWrap.load_from_checkpoint(ckpt_path + cfg.model_to_load,
                                                    # unet-best-checkpoint-2d-epoch=99-with-hela-bs-32.ckpt",
                                                    model_cfg=cfg.model, criterion=criterion,
                                                    optimizer_params=cfg.optimizer,
                                                    scheduler_params=cfg.scheduler if hasattr(cfg,
                                                                                              'scheduler') else None,
                                                    batch_size=cfg.data.batch_size, pre_trained_path=None)
            

        else:
            model = Unet2dWrap(cfg.model, criterion=criterion, optimizer_params=cfg.optimizer,
                               scheduler_params=cfg.scheduler if hasattr(cfg, 'scheduler') else None,
                               batch_size=cfg.data.batch_size, pre_trained_path=pre_trained_path)

        filename_ckpt = 'unet-best-checkpoint-2d-{epoch:03d}-new-loss-learned-alpha-start-at-4-beta-start-at-0-tanh-l1-l2-with-RMHD-1e-1-LMHD-1-seed-1051' #-focal-3

    elif cfg._2_5d:
        data_module = CELL3DDataModule(cfg.data)

        # cfg.data.img_size = cfg.data.img_size[-2:]
        if args.test:
            print(f"model_to_load: {cfg.model_to_load}")
            model = Unet2_5dWrap.load_from_checkpoint(ckpt_path + cfg.model_to_load,
                                                    # unet-best-checkpoint-2d-epoch=99-with-hela-bs-32.ckpt",
                                                    model_cfg=cfg.model, criterion=criterion,
                                                    optimizer_params=cfg.optimizer,
                                                    scheduler_params=cfg.scheduler if hasattr(cfg,
                                                                                              'scheduler') else None,
                                                    batch_size=cfg.data.batch_size, pre_trained_path=pre_trained_path)

        else:
            model = Unet2_5dWrap(cfg.model, criterion=criterion, optimizer_params=cfg.optimizer,
                               scheduler_params=cfg.scheduler if hasattr(cfg, 'scheduler') else None,
                               batch_size=cfg.data.batch_size, pre_trained_path=pre_trained_path)

            # model = Unet2dWrap.load_from_checkpoint("/raid/data/users/thomasm/ckpts/vit_unetbest-checkpoint-epoch=59-val_loss=6071.33.ckpt",
            #                                          model_cfg=cfg.model, criterion=criterion, optimizer_params=cfg.optimizer,
            #                                          scheduler_params=cfg.scheduler,
            #                                          batch_size=cfg.data.batch_size,
            #                                          pre_trained_path=None)
        filename_ckpt = 'unet2.5d-best-checkpoint-2-5d-{epoch:03d}-16-256-256-with-modify-op2'
        # filename_ckpt = '-best-checkpoint-2-5d-{epoch:03d}-16-128-128-with-modify-first-conv-op1'
    else:
        data_module = CELL3DDataModule(cfg.data)
        model = Unet3dWrap(cfg.model, criterion=criterion, optimizer_params=cfg.optimizer,
                           scheduler_params=cfg.scheduler if hasattr(cfg, 'scheduler') else None,
                           batch_size=cfg.data.batch_size, pre_trained_path=pre_trained_path)

        # model = Unet3dWrap.load_from_checkpoint("/raid/data/users/thomasm/ckpts/vit_unet/CHO-/vit_unet-best-checkpoint-3d-epoch=29.ckpt",
        #                                          model_cfg=cfg.model, criterion=criterion, optimizer_params=cfg.optimizer,
        #                                          scheduler_params=cfg.scheduler,
        #                                          batch_size=cfg.data.batch_size,
        #                                          pre_trained_path=None)
        filename_ckpt = 'unet3d-best-checkpoint-3d-{epoch:03d}-16-128-128-all-mhd'

    early_stopping_clbk = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=True)
    ckpt_clbk = ModelCheckpoint(
        monitor='val_SEG',  # Metric to monitor
        # monitor='train_SEG',  # Metric to monitor
        dirpath=ckpt_path,  # Directory to save checkpoints
        filename=filename_ckpt,  # Naming the saved checkpoint
        save_top_k=2,  # Save the top model based on val_loss
        mode='max',  # Minimize the val_loss
        every_n_epochs=5,
        save_last=False,  # Save the last model checkpoint for resuming
    )
    ckpt_last = ModelCheckpointSaveLast(
        dirpath=ckpt_path,
        # filename=cfg.name + filename_ckpt,
        filename=filename_ckpt,
    )
    lr_clbk = LearningRateMonitor(logging_interval='epoch')
    if cfg.data.sys == 'CTC':
        trainer_cfg = dict(
            accelerator='gpu' if torch.cuda.is_available() else "cpu",
            devices='auto',
            precision='16-mixed',
            logger=logger,
            default_root_dir=ckpt_path,
            **cfg.trainer
        )
    else:
        trainer_cfg = dict(
            accelerator='gpu',
            devices=devices,
            # strategy=strategy,
            precision='16-mixed',
            # max_epochs=cfg.trainer.pop('epochs'),
            callbacks=[ckpt_clbk, ckpt_last, lr_clbk],
            # callbacks=[ckpt_clbk, ckpt_last, lr_clbk, early_stopping_clbk],
            logger=logger,
            default_root_dir=ckpt_path,
            # log_every_n_steps=10,
            **cfg.trainer
        )

    if args.debug:
        trainer = pl.Trainer(fast_dev_run=True)
    else:
        trainer = pl.Trainer(**trainer_cfg)
    if args.test:
        trainer.test(model=model, datamodule=data_module, )
    else:
        trainer.fit(model=model, datamodule=data_module, )


