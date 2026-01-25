import os
import sys
import cv2
import math
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
import wandb
import numpy as np
from io import BytesIO
import matplotlib
import tifffile as tiff
from easydict import EasyDict
from scipy.optimize import linear_sum_assignment


matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.interpolation'] = 'none'
from scipy.ndimage import distance_transform_edt
from .base_model import BaseModel
import torch.nn.functional as F

from .blocks import DoubleConv2D, Up2DBlock, Up2DBlock2, DenseUp2DBlock, Dense2DBlock, DoubleConv3D, Up3DBlock, \
    DenseUp3DBlock, Dense3DBlock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import convert_3d_to_2_5d, convert_2_5d_to_3d, check_accuracy, plot_3d_images, plot_2d_images, \
    plot_2d_images_2, check_accuracy_CE, watershed_labels_from_binary, get_cell_instances, FocalBCEWithLogits

class Unet3d(nn.Module):
    def __init__(self, model_cfg, pre_trained_path=None):
        super(Unet3d, self).__init__()
        features = model_cfg.pop('features')
        in_channels = model_cfg.pop('in_channels')
        out_channels = model_cfg.pop('out_channels')
        self.signDistUps = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        for feature in reversed(features[1:-1]):
            self.signDistUps.append(Up3DBlock(feature * 2, feature // 2, feature))
        self.signDistUps.append(Up3DBlock(features[0] * 2, features[0], features[0], final_layer=True))
        self.signDistUps.append(nn.Conv3d(features[0], out_channels, kernel_size=1))

        self.bottleneck = Up3DBlock(features[-1], features[-2], features[-1])

        self.alpha = nn.Parameter(torch.tensor([1.3]))
        self.beta = nn.Parameter(torch.tensor([-0.3]))

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x_signDist = self.bottleneck(skip_connections.pop(), None)

        for signDistUp, skip_connection in zip(self.signDistUps[:-2], reversed(skip_connections[1:])):
            x_signDist = signDistUp(x_signDist, skip_connection)

        x_signDist = self.signDistUps[-2](x_signDist, skip_connections[0])
        x_signDist = self.signDistUps[-1](x_signDist)

        x_binSeg = self.alpha * x_signDist + self.beta
        return x_signDist, x_binSeg


class DenseUnet3d(nn.Module):
    def __init__(self, model_cfg, pre_trained_path=None):
        super(DenseUnet3d, self).__init__()

        features = model_cfg.pop('features')
        in_channels = model_cfg.pop('in_channels')
        out_channels = model_cfg.pop('out_channels')
        self.num_layers = model_cfg.pop('dense_num_layers')
        self.with_markers = model_cfg.pop('with_markers')
        self.signDistUps = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(Dense3DBlock(in_channels, feature, self.num_layers))
            in_channels = feature

        for feature in reversed(features[1:-1]):
            self.signDistUps.append(DenseUp3DBlock(feature * 2, feature // 2))
        self.signDistUps.append(DenseUp3DBlock(features[0] * 2, features[0] // 2, final_layer=True))
        self.signDistUps.append(nn.Conv3d(features[0] // 2, out_channels, kernel_size=1))

        self.bottleneck = DenseUp3DBlock(features[-1], features[-2])

        if self.with_markers:
            self.markersUps = nn.ModuleList()
            for feature in reversed(features[1:-1]):
                self.markersUps.append(DenseUp3DBlock(feature * 2, feature // 2))
            self.markersUps.append(DenseUp3DBlock(features[0] * 2, features[0] // 2, final_layer=True))
            self.markersUps.append(nn.Conv3d(features[0] // 2, 1, kernel_size=1))

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x_signDist = self.bottleneck(skip_connections.pop(), None)

        if self.with_markers:
            x_marker = x_signDist.clone()
            for signDistUp, markerUp, skip_connection in zip(self.signDistUps[:-1], self.markersUps[:-1],
                                                             reversed(skip_connections)):
                x_signDist, x_marker = signDistUp(x_signDist, skip_connection), markerUp(x_marker, skip_connection)
            return self.signDistUps[-1](x_signDist), self.markersUps[-1](x_marker)

        for signDistUp, skip_connection in zip(self.signDistUps[:-1], reversed(skip_connections)):
            x_signDist = signDistUp(x_signDist, skip_connection)
        return self.signDistUps[-1](x_signDist), None


class Unet2d(nn.Module):
    def __init__(self, model_cfg, pre_trained_path=None):
        super(Unet2d, self).__init__()
        features = model_cfg.pop('features')
        in_channels = model_cfg.pop('in_channels')
        out_channels = model_cfg.pop('out_channels')
        self.with_markers = model_cfg.pop('with_markers')
        self.signDistUps = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv2D(in_channels, feature))
            in_channels = feature

        for feature in reversed(features[1:-1]):
            self.signDistUps.append(Up2DBlock(feature * 2, feature // 2, feature))
        self.signDistUps.append(Up2DBlock(features[0] * 2, features[0], features[0], final_layer=True))
        self.signDistUps.append(nn.Conv2d(features[0], out_channels, kernel_size=1))
        # self.signDistOut = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.bottleneck = Up2DBlock(features[-1], features[-2], features[-1])

        if self.with_markers:
            self.binSeg = nn.ModuleList()
            self.binSeg.append(Up2DBlock(features[0] * 2, features[0], features[0], final_layer=True))
            self.binSeg.append(nn.Conv2d(features[0], out_channels, kernel_size=1))

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x_signDist = self.bottleneck(skip_connections.pop(), None)

        for signDistUp, skip_connection in zip(self.signDistUps[:-2], reversed(skip_connections[1:])):
            x_signDist = signDistUp(x_signDist, skip_connection)

        if self.with_markers:
            x_signDist, x_binSeg = (self.signDistUps[-2](x_signDist, skip_connections[0]),
                                    self.binSeg[0](x_signDist, skip_connections[0]))
            return self.signDistUps[-1](x_signDist), self.binSeg[1](x_binSeg)

        x_signDist = self.signDistUps[-2](x_signDist, skip_connections[0])
        return self.signDistUps[-1](x_signDist), None


class Unet2d_One_Leg(nn.Module):
    def __init__(self, model_cfg, pre_trained_path=None):
        super(Unet2d_One_Leg, self).__init__()
        features = model_cfg.pop('features')
        in_channels = model_cfg.pop('in_channels')
        out_channels = model_cfg.pop('out_channels')
        self.signDistUps = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv2D(in_channels, feature))
            in_channels = feature

        for feature in reversed(features[1:-1]):
            self.signDistUps.append(Up2DBlock(feature * 2, feature // 2, feature))
        self.signDistUps.append(Up2DBlock(features[0] * 2, features[0], features[0], final_layer=True))
        self.signDistUps.append(nn.Conv2d(features[0], out_channels, kernel_size=1))

        self.bottleneck = Up2DBlock(features[-1], features[-2], features[-1])
        # if alpha and beta are learned:
        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.beta = nn.Parameter(torch.tensor([0.0]))
        # if alpha and beta are fixed:
        # self.register_buffer("alpha", torch.tensor([4.0]))
        # self.register_buffer("beta", torch.tensor([0.0]))

        self.log_sigma2_mse = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2_mse_bin = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2_bce = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2_mhd_gt = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2_mhd_pred = nn.Parameter(torch.tensor(1.0))  # Suppress at the beginning

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x_signDist = self.bottleneck(skip_connections.pop(), None)

        for signDistUp, skip_connection in zip(self.signDistUps[:-2], reversed(skip_connections[1:])):
            x_signDist = signDistUp(x_signDist, skip_connection)

        x_signDist = self.signDistUps[-2](x_signDist, skip_connections[0])
        x_signDist = self.signDistUps[-1](x_signDist)

        x_binSeg = self.alpha * x_signDist + self.beta
        return x_signDist, x_binSeg


class DenseUnet2d(nn.Module):
    def __init__(self, model_cfg, pre_trained_path=None):
        super(DenseUnet2d, self).__init__()
        features = model_cfg.pop('features')
        in_channels = model_cfg.pop('in_channels')
        out_channels = model_cfg.pop('out_channels')
        self.num_layers = model_cfg.pop('dense_num_layers')
        self.with_markers = model_cfg.pop('with_markers')
        self.signDistUps = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(Dense2DBlock(in_channels, feature, self.num_layers))
            in_channels = feature

        for feature in reversed(features[1:-1]):
            self.signDistUps.append(DenseUp2DBlock(feature * 2, feature // 2, feature, self.num_layers))
        self.signDistUps.append(
            DenseUp2DBlock(features[0] * 2, features[0], features[0], self.num_layers, final_layer=True))
        self.signDistUps.append(nn.Conv2d(features[0], out_channels, kernel_size=1))

        self.bottleneck = DenseUp2DBlock(features[-1], features[-2], features[-1], self.num_layers)

        if self.with_markers:
            self.binSeg = nn.ModuleList()
            self.binSeg.append(
                DenseUp2DBlock(features[0] * 2, features[0], features[0], self.num_layers, final_layer=True))
            self.binSeg.append(nn.Conv2d(features[0], out_channels, kernel_size=1))
            # for feature in reversed(features[1:-1]):
            #     self.markersUps.append(DenseUp2DBlock(feature * 2, feature // 2, feature, self.num_layers))
            # self.markersUps.append(DenseUp2DBlock(features[0] * 2, features[0], features[0], self.num_layers, final_layer=True))
            # self.markersUps.append(nn.Conv2d(features[0], 1, kernel_size=1))

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x_signDist = self.bottleneck(skip_connections.pop(), None)

        for signDistUp, skip_connection in zip(self.signDistUps[:-2], reversed(skip_connections[1:])):
            x_signDist = signDistUp(x_signDist, skip_connection)

        if self.with_markers:
            x_signDist, x_binSeg = (self.signDistUps[-2](x_signDist, skip_connections[0]),
                                    self.binSeg[0](x_signDist, skip_connections[0]))
            return self.signDistUps[-1](x_signDist), self.binSeg[1](x_binSeg)

        x_signDist = self.signDistUps[-2](x_signDist, skip_connections[0])
        return self.signDistUps[-1](x_signDist), None
        # if self.with_markers:
        #     x_marker = x_signDist.clone()
        #     for signDistUp, markerUp, skip_connection in zip(self.signDistUps[:-1], self.markersUps[:-1],
        #                                                      reversed(skip_connections)):
        #         x_signDist, x_marker = signDistUp(x_signDist, skip_connection), markerUp(x_marker, skip_connection)
        #     return self.signDistUps[-1](x_signDist), self.markersUps[-1](x_marker)
        #
        # for signDistUp, skip_connection in zip(self.signDistUps[:-1], reversed(skip_connections)):
        #     x_signDist = signDistUp(x_signDist, skip_connection)
        # return self.signDistUps[-1](x_signDist), None

class Unet2_5d_Adapter(nn.Module):
    def __init__(self, model_cfg, pre_trained_path=None):
        super(Unet2_5d_Adapter, self).__init__()
        in_channels = model_cfg.in_channels
        model_cfg.in_channels = 1
        self.model = Unet2d_One_Leg(model_cfg)

        self.adapter_layer = nn.Conv2d(in_channels, 1, kernel_size=1)
        if pre_trained_path is not None:
            ckpt = torch.load(pre_trained_path, map_location='cpu')
            state_dict = ckpt["state_dict"]

            clean_state_dict = {
                k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")
            }
            # self.model.load_state_dict(clean_state_dict, strict=False)

            missing, unexpected = self.model.load_state_dict(clean_state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

    def forward(self,x):
        x = self.adapter_layer(x)
        return self.model(x)


class Unet2_5d_Modify(nn.Module):
    def __init__(self, model_cfg, pre_trained_path=None):
        super(Unet2_5d_Modify, self).__init__()
        in_channels = model_cfg.in_channels
        model_cfg.in_channels = 1
        self.model = Unet2d_One_Leg(model_cfg)
        if pre_trained_path is not None:
            ckpt = torch.load(pre_trained_path, map_location='cpu')
            state_dict = ckpt["state_dict"]

            clean_state_dict = {
                k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")
            }
            # self.model.load_state_dict(clean_state_dict, strict=False)

            missing, unexpected = self.model.load_state_dict(clean_state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        original_conv = self.model.downs[0].conv[0]
        if original_conv.in_channels != in_channels:
            new_conv = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )

            with torch.no_grad():
                # option 1 only the middle channel get the pre-trained weight
                new_conv.weight[:, 0:1, :, :] = 0.0
                new_conv.weight[:, 1:2, :, :] = original_conv.weight
                new_conv.weight[:, 2:, :, :] = 0.0
                # option 2 all channels get the pre--trained weight
                # for i in range(in_channels):
                #     new_conv.weight[:, i:i + 1, :, :] = original_conv.weight
            self.model.downs[0].conv[0] = new_conv

    def forward(self, x):
        return self.model(x)

class Unet3dWrap(BaseModel):
    def __init__(self, model_cfg, criterion, optimizer_params, scheduler_params, batch_size,
                 pre_trained_path=None):
        super().__init__(criterion, optimizer_params, scheduler_params)

        self.batch_size = batch_size
        self.alpha_mhd_gt = model_cfg.pop('alpha_mhd')
        self.alpha_mhd_pred = 0
        self.alpha_mse = model_cfg.pop('alpha_mse')
        self.sign_dist_leg_warmup = model_cfg.pop('sign_dist_leg_warmup')
        self.l1_lambda = model_cfg.pop('l1_lambda') if 'l1_lambda' in model_cfg else None

        if model_cfg.pop('dense'):
            self.model = DenseUnet3d(model_cfg=model_cfg, pre_trained_path=pre_trained_path)
        else:
            self.model = Unet3d(model_cfg=model_cfg, pre_trained_path=pre_trained_path)

        self.preds_signDist_list = []
        self.preds_bin_list = []
        self.targets_list = []
        self.test_sign_dist_targets = []
        self.data_list = []
        self.datasets_list = []
        self.preds_signDist_list_train = []
        self.preds_bin_list_train = []
        self.targets_list_train = []
        self.datasets_list_train = []

    def signed_distance_function(self, tensor):
        """
        Computes the signed distance function (SDF) for a given tensor.

        Args:
            tensor (torch.Tensor): A multi-class tensor.

        Returns:
            torch.Tensor: A tensor of the same shape with the signed distances.
        """
        np_array = tensor.cpu().numpy().astype(np.bool_)

        sd = np.zeros_like(np_array, dtype=np.float32)
        for i in np.unique(np_array):
            if i == 0:
                sd -= np.maximum(distance_transform_edt(np_array == i) - 1,
                                 0)  # remove 1 to set the edges as 0
            else:
                sd += np.maximum(distance_transform_edt(np_array == i) - 1,
                                 0)  # remove 1 to set the edges as 0

        sd_tensor = torch.from_numpy(sd).float().to(tensor.device)
        sd_tensor = (2 * torch.sigmoid(self.model.alpha * sd_tensor)) - 1
        return sd_tensor

    def batch_signed_distance_function(self, tensor):
        """
        Applies the signed distance function to each 3D slice in the 5D tensor.

        Args:
            tensor (torch.Tensor): A 5D tensor of shape (batch_size, depth, height, width).

        Returns:
            torch.Tensor: A 5D tensor with the same shape, containing the SDF for each 3D slice.
        """
        batch_size = tensor.shape[0]
        sd_tensor = torch.zeros_like(tensor).to(tensor.device)

        for b in range(batch_size):
            sd_slice = self.signed_distance_function(tensor[b])
            sd_tensor[b] = sd_slice

        return sd_tensor

    def remove_touching_boundary(self, targets):
        """
        Removes 1-voxel-wide touching boundaries between labeled regions
        for a batch of 3D labeled volumes of shape [B, 1, D, H, W].
        """
        device = targets.device
        B, _, D, H, W = targets.shape

        # 3D 26-connected neighborhood kernel (excluding center voxel)
        kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=device)
        kernel[0, 0, 1, 1, 1] = 0  # set center to 0

        results = torch.zeros_like(targets)

        for b in range(B):
            target = targets[b, 0]  # shape: [D, H, W]
            unique_labels = torch.unique(target)
            edge_mask = torch.zeros_like(target, dtype=torch.bool)

            for current_label in unique_labels:
                if current_label == 0:
                    continue

                binary = (target == current_label).float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                padded = F.pad(binary, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
                dilated = F.conv3d(padded, kernel)  # [1, 1, D, H, W]
                neighbors = dilated[0, 0]  # [D, H, W]

                # Touching if neighbors overlap with other labels
                touching = (neighbors > 0) & (target != current_label) & (target != 0)
                edge_mask |= touching

            cleaned = target.clone()
            cleaned[edge_mask] = 0
            results[b, 0] = cleaned

        return results

    def calcL1Loss(self):
        l1_loss = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
        return self.l1_lambda * l1_loss

    def modified_hausdorff_distance(self, sign_dist, edge_mask):
        return torch.sum(torch.abs(sign_dist * edge_mask)) / float(torch.sum(edge_mask).item())

    def detect_edges_sigmoid(self, bin_seg_pred):
        sig_bin_seg_pred = torch.sigmoid(bin_seg_pred)
        edge_mask = sig_bin_seg_pred * (1 - sig_bin_seg_pred)

        batch_min = edge_mask.amin(dim=(-3, -2, -1), keepdim=True)
        batch_max = edge_mask.amax(dim=(-3, -2, -1), keepdim=True)

        normalized_edge_mask = (edge_mask - batch_min) / (batch_max - batch_min + 1e-8)
        return normalized_edge_mask

    def calc_loss(self, sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets):
        l1_loss = 0
        mse_loss = 0
        bce_loss = 0
        mse_bin_loss = 0
        mhd_loss_pred = 0
        mhd_loss_gt = 0
        for loss_fn in self.criterion:
            if isinstance(loss_fn, nn.MSELoss) or isinstance(loss_fn, nn.SmoothL1Loss) or isinstance(loss_fn,
                                                                                                     nn.HuberLoss):
                mse_loss = self.alpha_mse * loss_fn(sign_dist_pred,
                                                    sign_dist_targets)

                mse_bin_loss = 5 * loss_fn(torch.sigmoid(bin_seg_pred), bin_targets)
            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                bce_loss = loss_fn(bin_seg_pred, bin_targets)

        mhd_loss_gt = self.alpha_mhd_gt * self.modified_hausdorff_distance(sign_dist_pred, (sign_dist_targets == 0))

        mhd_loss_pred = self.alpha_mhd_pred * self.modified_hausdorff_distance(sign_dist_targets,
                                                                              self.detect_edges_sigmoid(bin_seg_pred)
                                                                               )
        if self.l1_lambda is not None:
            l1_loss = self.calcL1Loss()  # start after getting the loss from sign dist leg

        return (mse_loss + bce_loss + mhd_loss_gt + mhd_loss_pred + l1_loss + mse_bin_loss, mse_loss, bce_loss,
                mhd_loss_gt, mhd_loss_pred, l1_loss, mse_bin_loss)

    def get_scores_by_dataset(self, datasets, seg_scores):
        scores_by_dataset = EasyDict()
        for ds, score in zip(datasets, seg_scores):
            if ds not in scores_by_dataset:
                scores_by_dataset[ds] = []
            scores_by_dataset[ds].append(score)
        return scores_by_dataset

    def prepare_batch(self, batch):
        data, original_targets, _ = batch

        # chose
        # targets = original_targets
        targets = self.remove_touching_boundary(original_targets)

        sign_dist_targets = self.batch_signed_distance_function(targets)
        bin_targets = (targets > 0).float()

        return data, sign_dist_targets, bin_targets, original_targets

    def predict_batch(self, batch):
        data, sign_dist_targets, bin_targets, original_targets = self.prepare_batch(batch)
        sign_dist_pred, bin_seg_pred = self.model(data)
        return sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets, original_targets

    def on_train_epoch_start(self):
        if self.current_epoch == self.sign_dist_leg_warmup:
            self.alpha_mhd_pred = self.alpha_mhd_gt
            # reset scheduler
            if hasattr(self, 'reduce_lr_scheduler'):
                if isinstance(self.reduce_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.reduce_lr_scheduler.best = self.reduce_lr_scheduler.mode_worse
                    self.reduce_lr_scheduler.cooldown_counter = 0
                    self.reduce_lr_scheduler.num_bad_epochs = 0
        elif self.current_epoch < self.sign_dist_leg_warmup and self.current_epoch % 10 == 0: #todo uncoment when not on harvard
            if hasattr(self, 'reduce_lr_scheduler'):
                if isinstance(self.reduce_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.reduce_lr_scheduler.best = self.reduce_lr_scheduler.mode_worse
                    self.reduce_lr_scheduler.cooldown_counter = 0
                    self.reduce_lr_scheduler.num_bad_epochs = 0


    def training_step(self, batch, batch_idx):
        sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets, original_targets = self.predict_batch(batch)
        if (self.current_epoch + 1) % 5 == 0:
            self.preds_signDist_list_train.append(sign_dist_pred)
            self.preds_bin_list_train.append(bin_seg_pred)
            self.targets_list_train.append(original_targets)
            self.datasets_list_train += batch[-1]
        loss, mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, l1_loss, mse_bin_loss = self.calc_loss(
            sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_mse_loss', mse_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_mse_bin_loss', mse_bin_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_bce_loss', bce_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_mhd_loss SD_pred*(gt==0)', mhd_loss_gt, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_mhd_loss (pred==0)*SD_GT', mhd_loss_pred, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_l1_loss', l1_loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 5 == 0:
            print(f"\nCalculating train Jaccard Index for epoch: {self.current_epoch}")

            seg_scores, det_score = check_accuracy(self.preds_signDist_list_train, self.preds_bin_list_train,
                                        self.targets_list_train,
                                        batch_size=self.batch_size, three_d=True, _2_5d=False)

            self.log("train_SEG", seg_scores.mean(), on_epoch=True, prog_bar=True)
            print(f"total train SEG: mean={seg_scores.mean():.3f}, std={seg_scores.std():.3f}")
            scores_by_dataset = self.get_scores_by_dataset(self.datasets_list_train, seg_scores)
            for ds, scores in scores_by_dataset.items():
                scores = np.array(scores)
                mean = scores.mean()
                std = scores.std()
                print(f"{ds}: mean={mean:.3f}, std={std:.3f}")
                self.log(f"train_SEG_{ds}", mean, on_epoch=True, prog_bar=True)
            self.preds_signDist_list_train.clear()
            self.preds_bin_list_train.clear()
            self.targets_list_train.clear()
            self.datasets_list_train.clear()

    def validation_step(self, batch, batch_idx):
        sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets, original_targets = self.predict_batch(batch)
        if (self.current_epoch + 1) % 5 == 0:
            self.preds_signDist_list.append(sign_dist_pred)
            self.preds_bin_list.append(bin_seg_pred)
            self.targets_list.append(original_targets)
            self.datasets_list += batch[-1]
            if not self.test_sign_dist_targets:
                self.test_sign_dist_targets.append(sign_dist_targets)
                self.data_list.append(batch[0])
        loss, mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, l1_loss, mse_bin_loss = self.calc_loss(
            sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets)
        self.log("val_loss", loss, batch_size=batch[0].shape[0], on_epoch=True, on_step=False)

        self.log('val_mse_loss', mse_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_mse_bin_loss', mse_bin_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_bce_loss', bce_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_mhd_loss SD_pred*(gt==0)', mhd_loss_gt, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_mhd_loss (pred==0)*SD_GT', mhd_loss_pred, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_l1_loss', l1_loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_end(self):
        self.log("alpha", self.model.alpha.item(), prog_bar=False)
        self.log("beta", self.model.beta.item(), prog_bar=False)

        if (self.current_epoch + 1) % 5 == 0:
            print(f"\nCalculating val Jaccard Index for epoch: {self.current_epoch}")

            seg_scores, det_scores = check_accuracy(self.preds_signDist_list, self.preds_bin_list, self.targets_list,
                                        batch_size=self.batch_size, three_d=True, _2_5d=False)
            self.log("val_SEG", seg_scores.mean(), on_epoch=True, prog_bar=True)
            print(f"total val SEG: mean={seg_scores.mean():.3f}, std={seg_scores.std():.3f}")
            scores_by_dataset = self.get_scores_by_dataset(self.datasets_list, seg_scores)
            for ds, scores in scores_by_dataset.items():
                scores = np.array(scores)
                mean = scores.mean()
                std = scores.std()
                print(f"{ds}: mean={mean:.3f}, std={std:.3f}")
                self.log(f"val_SEG_{ds}", mean, on_epoch=True, prog_bar=True)
            middle_slice = self.data_list[0].shape[2] // 2
            input_image = self.data_list[0][0, 0, middle_slice].cpu().numpy()
            sd_image = self.preds_signDist_list[0][0, 0, middle_slice].cpu().numpy()
            gt_image = self.targets_list[0][0, 0, middle_slice].cpu().numpy()

            edge_map = self.detect_edges_sigmoid(self.preds_bin_list[0])

            bin_pred_hard_0_5 = (torch.sigmoid(self.preds_bin_list[0]) > 0.5)[0, 0, middle_slice].cpu().numpy()
            bin_pred_soft = torch.sigmoid(self.preds_bin_list[0])[0, 0, middle_slice].cpu().numpy()
            inv_bin_pred_soft = 1 - bin_pred_soft
            edge_map_np = edge_map[0, 0, middle_slice].cpu().numpy()

            instance_pred, _ = get_cell_instances(bin_pred_hard_0_5, three_d=False)
            instance_pred[instance_pred != 0] += 5

            fig, axs = plt.subplots(2, 5, figsize=(15, 5))
            axs[0, 0].imshow(input_image, cmap='gray')
            axs[0, 0].set_title("Input Image")
            axs[0, 0].axis('off')

            im0 = axs[0, 1].imshow(sd_image, cmap='jet')
            axs[0, 1].set_title("SD Prediction")
            axs[0, 1].axis('off')
            plt.colorbar(im0, ax=axs[0, 1], orientation='vertical', fraction=0.046, pad=0.04)

            im1 = axs[0, 2].imshow(bin_pred_soft, cmap='jet')
            axs[0, 2].set_title("Soft Binary Prediction")
            axs[0, 2].axis('off')
            plt.colorbar(im1, ax=axs[0, 2], orientation='vertical', fraction=0.046, pad=0.04)

            axs[0, 3].imshow(bin_pred_hard_0_5, cmap='viridis')
            axs[0, 3].set_title(f"Binary Prediction > 0.5")
            axs[0, 3].axis('off')

            axs[0, 4].imshow(instance_pred, cmap='nipy_spectral')
            axs[0, 4].set_title(f"Instance Prediction")
            axs[0, 4].axis('off')

            im1 = axs[1, 0].imshow(inv_bin_pred_soft, cmap='jet')
            axs[1, 0].set_title("Inverse Soft Binary Prediction")
            axs[1, 0].axis('off')
            plt.colorbar(im1, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)

            im2 = axs[1, 1].imshow(edge_map_np, cmap='jet')
            axs[1, 1].set_title("Prediction Edges Map")
            axs[1, 1].axis('off')
            plt.colorbar(im2, ax=axs[1, 1], orientation='vertical', fraction=0.046, pad=0.04)

            im3 = axs[1, 2].imshow(self.test_sign_dist_targets[0][0, 0, middle_slice].cpu().numpy(), cmap='jet')
            axs[1, 2].set_title("SD Ground Truth")
            axs[1, 2].axis('off')
            plt.colorbar(im3, ax=axs[1, 2], orientation='vertical', fraction=0.046, pad=0.04)

            axs[1, 3].imshow(gt_image > 0, cmap='viridis')
            axs[1, 3].set_title("Semantic Ground Truth")
            axs[1, 3].axis('off')

            axs[1, 4].imshow(gt_image, cmap='nipy_spectral')
            axs[1, 4].set_title("Ground Truth")
            axs[1, 4].axis('off')

            plt.tight_layout()

            self.trainer.logger.experiment.log({"image": wandb.Image(fig)})
            plt.close(fig)

            self.preds_signDist_list.clear()
            self.preds_bin_list.clear()
            self.targets_list.clear()
            self.test_sign_dist_targets.clear()
            self.data_list.clear()
            self.datasets_list.clear()

    def test_step(self, batch, batch_idx):
        sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets, original_targets = self.predict_batch(batch)

        self.preds_signDist_list.append(sign_dist_pred)
        self.preds_bin_list.append(bin_seg_pred)
        self.targets_list.append(original_targets)
        self.test_sign_dist_targets.append(sign_dist_targets)
        self.data_list.append(batch[0])
        self.datasets_list += batch[-1]

    def on_test_epoch_end(self):
        print("******************************************************")  # todo
        print("todo")
        print("******************************************************")
        print(f"Calculating Jaccard Index")

        jaccard, std = check_accuracy(self.test_preds_signDist, self.test_preds_markers, self.test_targets,
                                      batch_size=self.batch_size, three_d=True, _2_5d=False)
        self.log("jaccard_index", jaccard, on_epoch=True, prog_bar=True)

        plot_3d_images([self.test_preds_signDist[0], self.test_sign_dist_targets[0], self.test_targets[0]])

        self.test_preds_signDist.clear()
        self.test_preds_markers.clear()
        self.test_targets.clear()
        self.test_sign_dist_targets.clear()


class Unet2dWrap(BaseModel):
    def __init__(self, model_cfg, criterion, optimizer_params, scheduler_params, batch_size, pre_trained_path):
        super().__init__(criterion, optimizer_params, scheduler_params)

        self.batch_size = batch_size
        self.output_dir = model_cfg.pop('output_dir', None)
        self.sigmoid_alpha = model_cfg.pop('sigmoid_alpha')
        self.kernel_type = model_cfg.pop('kernel_type')
        self.alpha_mhd_gt = model_cfg.pop('alpha_mhd_gt')
        self.alpha_mhd_pred = model_cfg.pop('alpha_mhd_pred')
        self.alpha_bce = model_cfg.pop('alpha_bce')
        self.alpha_focal = model_cfg.pop('alpha_focal')
        self.alpha_mse = model_cfg.pop('alpha_mse')
        self.alpha_l1 = model_cfg.pop('alpha_l1')
        self.alpha_huber = model_cfg.pop('alpha_huber')
        self.one_leg_model = model_cfg.pop('one_leg')
        self.uaw_loss = model_cfg.pop('uaw_loss')
        # self.image_test_hela = tiff.imread(model_cfg.pop('test_hela_path'))
        # self.mask_test_hela = tiff.imread(model_cfg.pop('test_hela_mask_path'))
        # self.image_test_sim = tiff.imread(model_cfg.pop('test_sim_path'))
        # self.mask_test_sim = tiff.imread(model_cfg.pop('test_sim_mask_path'))
        if model_cfg.with_markers and 'sign_dist_leg_warmup' in model_cfg:
            self.sign_dist_leg_warmup = model_cfg.pop('sign_dist_leg_warmup')
        else:
            self.sign_dist_leg_warmup = 0
        if self.uaw_loss:
            self.alpha_mhd_gt = 1
            self.alpha_mhd_pred = 1

        self.l1_lambda = model_cfg.pop('l1_lambda') if 'l1_lambda' in model_cfg else None
        if model_cfg.pop('dense'):
            self.model = DenseUnet2d(model_cfg=model_cfg, pre_trained_path=pre_trained_path)
        elif self.one_leg_model:
            self.model = Unet2d_One_Leg(model_cfg=model_cfg, pre_trained_path=pre_trained_path)
        else:
            self.model = Unet2d(model_cfg=model_cfg, pre_trained_path=pre_trained_path)
        self.preds_signDist_list = []
        self.preds_bin_list = []
        self.targets_list = []
        self.test_sign_dist_targets = []
        self.data_list = []
        self.datasets_list = []
        self.preds_signDist_list_train = []
        self.preds_bin_list_train = []
        self.targets_list_train = []
        self.datasets_list_train = []
        self.mask_name_list = []
        # print(self.model)
        # print("--------------------lets play ----------------------------")
        # print(self.model.downs[0])
        # print("--------------------lets play 2----------------------------")
        # print(self.model.downs[0].conv[0])
        # print("--------------------lets play 3----------------------------")
        # mid_channels = self.model.downs[0].conv[0].out_channels
        # self.model.downs[0].conv[0] = nn.Conv2d(3, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # print(self.model)

    def signed_distance_function(self, tensor):
        """
        Computes the signed distance function (SDF) for a given tensor.

        Args:
            tensor (torch.Tensor): A multi-class tensor.

        Returns:
            torch.Tensor: A tensor of the same shape with the signed distances.
        """
        np_array = tensor.cpu().numpy().astype(np.bool_)

        # outside_distances = np.maximum(distance_transform_edt(np_array == 0) - 1, 0)  # remove 1 to set the edges as 0
        # inside_distances = np.maximum(distance_transform_edt(np_array > 0) - 1, 0)  # remove 1 to set the edges as 0
        #
        # # SD = inside distance (positive) - outside distance (negative)
        # sd = inside_distances - outside_distances
        sd = np.zeros_like(np_array, dtype=np.float32)
        for i in np.unique(np_array):
            if i == 0:
                sd -= np.maximum(distance_transform_edt(np_array == i) - 1,
                                 0)  # remove 1 to set the edges as 0
            else:
                sd += np.maximum(distance_transform_edt(np_array == i) - 1,
                                 0)  # remove 1 to set the edges as 0

        sd_tensor = torch.from_numpy(sd).float().to(tensor.device)
        # if self.one_leg_model:
        #     # sd_tensor = (2 * torch.sigmoid(self.model.alpha * sd_tensor + self.model.beta)) - 1
        #     # sd_tensor = (2 * torch.sigmoid(self.model.alpha * sd_tensor)) - 1
        #
        #     sd_tensor = self.model.alpha * sd_tensor + self.model.beta
        #     # sd_tensor = 0.1 * sd_tensor
        # else:
        #     sd_tensor = (2 * torch.sigmoid(self.sigmoid_alpha * sd_tensor)) - 1
        return sd_tensor

    def batch_signed_distance_function(self, tensor):
        """
        Applies the signed distance function to each 2D slice in the 4D tensor.

        Args:
            tensor (torch.Tensor): A 4D tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: A 4D tensor with the same shape, containing the SD for each 2D slice.
        """

        batch_size = tensor.shape[0]
        sd_tensor = torch.zeros_like(tensor).to(tensor.device)

        for b in range(batch_size):
            sd_slice = self.signed_distance_function(tensor[b])
            sd_tensor[b] = sd_slice

        return sd_tensor

    def remove_touching_boundary(self, targets):
        """
        Removes 1-pixel-wide touching boundaries between labeled regions
        for a batch of 2D labeled images of shape [B, 1, H, W].
        """
        device = targets.device
        B, _, H, W = targets.shape

        kernel = torch.tensor([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        results = torch.zeros_like(targets)

        for b in range(B):
            target = targets[b, 0]
            unique_labels = torch.unique(target)
            edge_mask = torch.zeros_like(target, dtype=torch.bool)

            for i, current_label in enumerate(unique_labels):
                if current_label == 0:
                    continue

                binary = (target == current_label).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                dilated = F.conv2d(F.pad(binary, (1, 1, 1, 1), mode='constant', value=0), kernel)
                neighbors = dilated[0, 0]  # [H, W]

                # Where dilation overlaps with *other* labels
                touching = (neighbors > 0) & (target != current_label) & (target != 0)
                edge_mask |= touching

            cleaned = target.clone()
            cleaned[edge_mask] = 0
            results[b, 0] = cleaned

        return results

    def calcL1Loss(self):
        l1_loss = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
        return self.l1_lambda * l1_loss

    def modified_hausdorff_distance(self, sign_dist, sd_to_edges):
        edge_mask = self.detect_edges_sigmoid(sd_to_edges)
        return torch.sum(torch.abs(sign_dist) * edge_mask) / float(torch.sum(edge_mask).item())

    def detect_edges_grad(self, pred, threshold_grad=0.2, threshold_sigmoid=0.5):
        bin_pred = (torch.sigmoid(self.sigmoid_alpha * pred) > threshold_sigmoid).float()
        gradient_width = torch.gradient(bin_pred, dim=-1)[0]  # Gradient along width
        gradient_height = torch.gradient(bin_pred, dim=-2)[0]  # Gradient along height

        gradient_magnitude = torch.sqrt(gradient_width ** 2 + gradient_height ** 2)
        masked_gradient_magnitude = gradient_magnitude * bin_pred
        edge_mask = (masked_gradient_magnitude > threshold_grad).to(torch.int).to(pred.device)

        return edge_mask

    def detect_edges_laplac(self, bin_seg_pred, threshold=0.1, kernel_type="smoothed"):
        """
        Detect edges using the Laplacian filter on a logit prediction.

        Args:
            bin_seg_pred (torch.Tensor): Logit predictions (after sigmoid).
            threshold (float): Threshold for edge detection.
            kernel_type (str): Type of Laplacian kernel to use ("standard", "diagonal", "smoothed").

        Returns:
            torch.Tensor: Binary edge mask.
        """

        laplacian_kernels = {
            "standard": torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32),
            "diagonal": torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32),
            "smoothed": torch.tensor([[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]], dtype=torch.float32)
        }

        # Select the desired kernel
        kernel = laplacian_kernels.get(kernel_type, laplacian_kernels["standard"])
        kernel = kernel.view(1, 1, 3, 3).to(bin_seg_pred.device, dtype=bin_seg_pred.dtype)  # Reshape for conv2d

        # Apply convolution with the Laplacian kernel
        edges = F.conv2d(bin_seg_pred, kernel, padding=1)
        edge_mask = (edges.abs() > threshold)
        return edge_mask

    def detect_edges_sigmoid(self, target):
        sig_target = torch.sigmoid(target)
        edge_mask = sig_target * (1 - sig_target)
        batch_min = edge_mask.amin(dim=(-2, -1), keepdim=True)
        batch_max = edge_mask.amax(dim=(-2, -1), keepdim=True)

        edge_mask = (edge_mask - batch_min) / (batch_max - batch_min + 1e-8)
        return edge_mask

    def calc_loss(self, sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets):
        l1_loss = 0
        mse_loss = 0
        bce_loss = 0
        bce_loss_of_sign_dist = 0
        for loss_fn in self.criterion:
            if isinstance(loss_fn, nn.MSELoss):
                mse_loss = self.alpha_mse * loss_fn(sign_dist_pred, sign_dist_targets)
            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                if bin_seg_pred is not None:
                    bce_loss = loss_fn(bin_seg_pred, bin_targets)
                bce_loss_of_sign_dist = self.sign_dist_leg * loss_fn(sign_dist_pred, bin_targets)
        mhd_loss_gt = self.mhd_scheduler * self.modified_hausdorff_distance(sign_dist_pred, (sign_dist_targets == 0))
        mhd_loss_pred = self.mhd_scheduler * self.modified_hausdorff_distance(sign_dist_targets,
                                                                              self.detect_edges_grad(sign_dist_pred))

        if self.l1_lambda is not None:
            l1_loss = self.sign_dist_leg * self.calcL1Loss()  # start after getting the loss from sign dist leg

        return (mse_loss + bce_loss + bce_loss_of_sign_dist + mhd_loss_gt + mhd_loss_pred + l1_loss, mse_loss, bce_loss,
                bce_loss_of_sign_dist, mhd_loss_gt, mhd_loss_pred, l1_loss)
        # return mse_loss + bce_loss + bce_loss_of_sign_dist + mhd_loss_gt + mhd_loss_pred + l1_loss

    def calc_loss_one_leg(self, sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets):
        l1_loss = 0
        mse_loss = 0
        bce_loss = 0
        mse_bin_loss = 0
        mhd_loss_pred = 0
        mhd_loss_gt = 0
        sig_bin_seg_pred = torch.sigmoid(bin_seg_pred)
        for loss_fn in self.criterion:
            if isinstance(loss_fn, nn.MSELoss) or isinstance(loss_fn, nn.SmoothL1Loss) or isinstance(loss_fn, nn.HuberLoss):
                mse_loss = self.alpha_mse * loss_fn(sign_dist_pred,
                                                    sign_dist_targets)  # torch.sigmoid(bin_seg_pred)

                # mse_bin_loss = 5 * loss_fn(torch.sigmoid(bin_seg_pred), (sign_dist_targets + 1) / 2)
                mse_bin_loss = 5 * loss_fn(torch.sigmoid(bin_seg_pred), bin_targets)
            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                # bce_loss = loss_fn(bin_seg_pred, (sign_dist_targets + 1) / 2)
                bce_loss = loss_fn(bin_seg_pred, bin_targets)

        mhd_loss_gt = self.alpha_mhd_gt * self.modified_hausdorff_distance(sign_dist_pred, (
                sign_dist_targets == 0))  # sig_bin_seg_pred

        if self.kernel_type is None:
            mhd_loss_pred = self.alpha_mhd_pred * self.modified_hausdorff_distance(sign_dist_targets,
                                                                                   self.detect_edges_grad(
                                                                                       bin_seg_pred))
        elif self.kernel_type == 'sigmoid':
            mhd_loss_pred = self.alpha_mhd_pred * self.modified_hausdorff_distance(sign_dist_targets,
                                                                                       bin_seg_pred
                                                                                   )
            # mhd_loss_gt = self.alpha_mhd_gt * self.modified_hausdorff_distance(sign_dist_pred,
            #                                                                    self.detect_edges_sigmoid(
            #                                                                            sign_dist_targets)
            #                                                                        )
        else:
            mhd_loss_pred = self.alpha_mhd_pred * self.modified_hausdorff_distance(sign_dist_targets,
                                                                                   self.detect_edges_laplac(
                                                                                       sig_bin_seg_pred,
                                                                                       kernel_type=self.kernel_type))

        if self.l1_lambda is not None:
            l1_loss = self.calcL1Loss()  # start after getting the loss from sign dist leg

        return (mse_loss + bce_loss + mhd_loss_gt + mhd_loss_pred + l1_loss + mse_bin_loss, mse_loss, bce_loss,
                mhd_loss_gt, mhd_loss_pred, l1_loss, mse_bin_loss)

    def calc_loss_one_leg2(self, sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets):
        reg_l1_loss = 0
        mse_loss = 0
        bce_loss = 0
        focal_loss = 0
        l1_loss = 0
        huber_loss = 0
        mhd_loss_pred = 0
        mhd_loss_gt = 0
        sign_dist_targets_a_b = self.model.alpha * sign_dist_targets + self.model.beta
        tanh_pred = torch.tanh(bin_seg_pred)
        tanh_gt = torch.tanh(sign_dist_targets_a_b)
        for loss_fn in self.criterion:
            if isinstance(loss_fn, nn.MSELoss):
                mse_loss = self.alpha_mse * loss_fn(tanh_pred, tanh_gt)
                # mse_loss = self.alpha_mse * loss_fn(sign_dist_pred, sign_dist_targets)

            if isinstance(loss_fn, nn.L1Loss):
                l1_loss = self.alpha_l1 * loss_fn(tanh_pred, tanh_gt)
                # l1_loss = self.alpha_l1 * loss_fn(sign_dist_pred, sign_dist_targets)

            if isinstance(loss_fn, nn.HuberLoss):
                huber_loss = self.alpha_huber * loss_fn(tanh_pred, tanh_gt)
                # huber_loss = self.alpha_huber * loss_fn(sign_dist_pred, sign_dist_targets)

            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                bce_loss = self.alpha_bce * loss_fn(bin_seg_pred, bin_targets)

            if isinstance(loss_fn, FocalBCEWithLogits):
                focal_loss = self.alpha_focal * loss_fn(bin_seg_pred, bin_targets)

        mhd_loss_gt = self.alpha_mhd_gt * self.modified_hausdorff_distance(tanh_pred,
                                                                           sign_dist_targets_a_b
                                                                           )
        # temp_pred = self.modified_hausdorff_distance(tanh_gt, bin_seg_pred)
        # mhd_loss_pred = self.alpha_mhd_pred * temp_pred
        mhd_loss_pred = self.alpha_mhd_pred * self.modified_hausdorff_distance(tanh_gt,
                                                                               bin_seg_pred
                                                                               )



        if self.l1_lambda is not None:
            reg_l1_loss = self.calcL1Loss()  # start after getting the loss from sign dist leg

        return (mse_loss + bce_loss + mhd_loss_gt + reg_l1_loss + l1_loss + huber_loss + focal_loss + mhd_loss_pred,
                mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, reg_l1_loss, l1_loss, huber_loss, focal_loss)
        # return (mse_loss + bce_loss + mhd_loss_gt + reg_l1_loss + l1_loss + huber_loss + focal_loss,   # + mhd_loss_pred
        #         mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, reg_l1_loss, l1_loss, huber_loss, focal_loss)

    def calc_loss_one_leg_with_uaw(self, sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets):
        def uaw_loss(log_sigma2, loss_term):
            # s = torch.clamp(log_sigma2, min=-0.2, max=4.2)
            # s = torch.clamp(log_sigma2, min=-1.0, max=4.2)
            return 0.5 * torch.exp(-log_sigma2) * loss_term, 0.5 * log_sigma2
            # return 0.5 * torch.exp(-s) * loss_term, 0.5 * s

        l1_loss = 0
        mse_loss = 0
        bce_loss = 0
        mse_bin_loss = 0
        mhd_loss_pred = 0
        mhd_loss_gt = 0
        sig_bin_seg_pred = torch.sigmoid(bin_seg_pred)
        for loss_fn in self.criterion:
            if isinstance(loss_fn, nn.MSELoss) or isinstance(loss_fn, nn.SmoothL1Loss):
                mse_loss = self.alpha_mse * loss_fn(sign_dist_pred,
                                                    sign_dist_targets)  # torch.sigmoid(bin_seg_pred)

                # mse_bin_loss = loss_fn(torch.sigmoid(bin_seg_pred), (sign_dist_targets + 1) / 2)
                mse_bin_loss = loss_fn(torch.sigmoid(bin_seg_pred), bin_targets)
            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                # bce_loss = loss_fn(bin_seg_pred, (sign_dist_targets + 1) / 2)
                bce_loss = loss_fn(bin_seg_pred, bin_targets)

        mhd_loss_gt = self.alpha_mhd_gt * self.modified_hausdorff_distance(sign_dist_pred, (
                sign_dist_targets == 0))  # sig_bin_seg_pred

        if self.kernel_type is None:
            mhd_loss_pred = self.alpha_mhd_pred * self.modified_hausdorff_distance(sign_dist_targets,
                                                                                   self.detect_edges_grad(
                                                                                       bin_seg_pred))
        elif self.kernel_type == 'sigmoid':
            mhd_loss_pred = self.alpha_mhd_pred * self.modified_hausdorff_distance(sign_dist_targets,
                                                                                   self.detect_edges_sigmoid(
                                                                                       bin_seg_pred)
                                                                                   )
        else:
            mhd_loss_pred = self.alpha_mhd_pred * self.modified_hausdorff_distance(sign_dist_targets,
                                                                                   self.detect_edges_laplac(
                                                                                       sig_bin_seg_pred,
                                                                                       kernel_type=self.kernel_type))

        if self.l1_lambda is not None:
            l1_loss = self.calcL1Loss()  # start after getting the loss from sign dist leg
        mse_loss_uaw, reg_mse = uaw_loss(self.model.log_sigma2_mse, mse_loss)
        mse_bin_loss_uaw, reg_mse_bin = uaw_loss(self.model.log_sigma2_mse_bin, mse_bin_loss)
        bce_loss_uaw, reg_bce = uaw_loss(self.model.log_sigma2_bce, bce_loss)
        mhd_loss_gt_uaw, reg_mhd_gt = uaw_loss(self.model.log_sigma2_mhd_gt, mhd_loss_gt)
        mhd_loss_pred_uaw, reg_mhd_pred = uaw_loss(self.model.log_sigma2_mhd_pred, mhd_loss_pred)
        total_loss = (
                    mse_loss_uaw + mse_bin_loss_uaw + bce_loss_uaw + mhd_loss_gt_uaw + mhd_loss_pred_uaw + l1_loss  # )
                    + reg_mse + reg_mse_bin + reg_bce + reg_mhd_gt + reg_mhd_pred)

        return total_loss, mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, l1_loss, mse_bin_loss, mse_loss_uaw, mse_bin_loss_uaw, bce_loss_uaw, mhd_loss_gt_uaw, mhd_loss_pred_uaw

    def overlay_edges_on_grayscale(self, grayscale_img, pred_edge_map, gt_edge_map):
        """
        Overlays predicted and ground truth edge maps on the grayscale image.

        Args:
            grayscale_img (numpy array): Grayscale image of shape [H, W].
            pred_edge_map (numpy array): Predicted edge map of shape [H, W].
            gt_edge_map (numpy array): Ground truth edge map of shape [H, W].

        Returns:
            RGB image with edges overlayed
        """
        grayscale_img = ((grayscale_img - grayscale_img.min()) / (
                grayscale_img.max() - grayscale_img.min()) * 255).astype(np.uint8)

        rgb_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
        max_val = np.iinfo(grayscale_img.dtype).max
        rgb_img[pred_edge_map >= 0.85, 0] = max_val  # Set red channel
        rgb_img[gt_edge_map >= 0.85, 1] = max_val  # Set green channel

        return rgb_img

    def get_scores_by_dataset(self, datasets, seg_scores):
        scores_by_dataset = EasyDict()
        for ds, score in zip(datasets, seg_scores):
            if ds not in scores_by_dataset:
                scores_by_dataset[ds] = []
            scores_by_dataset[ds].append(score)
        return scores_by_dataset

    def stack_2d_labels_to_3d(self, label_slices, base_thresh=0.3, min_thresh=0.1):
        def compute_iou_matrix(curr, prev):
            curr_labels = np.unique(curr)
            prev_labels = np.unique(prev)
            curr_labels = curr_labels[curr_labels != 0]
            prev_labels = prev_labels[prev_labels != 0]

            iou_matrix = np.zeros((len(curr_labels), len(prev_labels)), dtype=np.float32)
            curr_sizes = []
            prev_sizes = []

            for i, c_label in enumerate(curr_labels):
                c_mask = curr == c_label
                c_area = c_mask.sum()
                curr_sizes.append(c_area)
                for j, p_label in enumerate(prev_labels):
                    p_mask = prev == p_label
                    if i == 0:  # only collect once
                        prev_sizes.append(p_mask.sum())
                    inter = np.logical_and(c_mask, p_mask).sum()
                    union = c_area + prev_sizes[j] - inter
                    iou_matrix[i, j] = inter / union if union > 0 else 0

            return iou_matrix, curr_labels, prev_labels, np.array(curr_sizes), np.array(prev_sizes)

        Z = len(label_slices)
        H, W = label_slices[0].shape
        label_volume = np.zeros((Z, H, W), dtype=np.int32)
        label_counter = 1
        label_map = {}

        for z in range(Z):
            print(f"[{z} / {Z - 1}]")
            curr = label_slices[z]
            new_slice = np.zeros_like(curr)

            if z == 0:
                for l in np.unique(curr):
                    if l == 0: continue
                    new_slice[curr == l] = label_counter
                    label_map[(z, l)] = label_counter
                    label_counter += 1
            else:
                prev = label_volume[z - 1]
                iou_matrix, curr_labels, prev_labels, curr_sizes, prev_sizes = compute_iou_matrix(curr, prev)
                cost_matrix = 1 - iou_matrix
                row_idx, col_idx = linear_sum_assignment(cost_matrix)

                matched_curr = set()
                for r, c in zip(row_idx, col_idx):
                    iou = iou_matrix[r, c]
                    size = curr_sizes[r]
                    # adaptive threshold
                    thresh = base_thresh if size >= 100 else min_thresh
                    if iou >= thresh:
                        c_label = curr_labels[r]
                        p_label = prev_labels[c]
                        consistent_id = np.unique(prev[prev == p_label])[0]
                        new_slice[curr == c_label] = consistent_id
                        label_map[(z, c_label)] = consistent_id
                        matched_curr.add(c_label)

                # assign new IDs to unmatched
                for c_label in curr_labels:
                    if c_label not in matched_curr:
                        new_slice[curr == c_label] = label_counter
                        label_map[(z, c_label)] = label_counter
                        label_counter += 1

            label_volume[z] = new_slice

        return label_volume

    def prepare_batch(self, batch):
        data, original_targets, _ = batch
        original_targets = original_targets.unsqueeze(1)

        # chose
        # targets = original_targets
        targets = self.remove_touching_boundary(original_targets)

        sign_dist_targets = self.batch_signed_distance_function(targets)
        bin_targets = (targets > 0).float()

        return data, sign_dist_targets, bin_targets, original_targets

    def predict_batch(self, batch):
        data, sign_dist_targets, bin_targets, original_targets = self.prepare_batch(batch)
        sign_dist_pred, bin_seg_pred = self.model(data)
        return sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets, original_targets

    def on_train_epoch_start(self):
        if not self.uaw_loss:
            if self.current_epoch == self.sign_dist_leg_warmup:
                # self.alpha_mhd_pred = self.alpha_mhd_gt
                # self.alpha_mhd_pred = 1
                # reset scheduler
                if hasattr(self, 'reduce_lr_scheduler'):
                    if isinstance(self.reduce_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.reduce_lr_scheduler.best = self.reduce_lr_scheduler.mode_worse
                        self.reduce_lr_scheduler.cooldown_counter = 0
                        self.reduce_lr_scheduler.num_bad_epochs = 0
            elif self.current_epoch < self.sign_dist_leg_warmup and self.current_epoch % 10 == 0: #todo uncoment when not on harvard
                if hasattr(self, 'reduce_lr_scheduler'):
                    if isinstance(self.reduce_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.reduce_lr_scheduler.best = self.reduce_lr_scheduler.mode_worse
                        self.reduce_lr_scheduler.cooldown_counter = 0
                        self.reduce_lr_scheduler.num_bad_epochs = 0


    def training_step(self, batch, batch_idx):
        sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets, original_targets = self.predict_batch(batch)
        if (self.current_epoch + 1) % 5 == 0:
            self.preds_signDist_list_train.append(sign_dist_pred)
            self.preds_bin_list_train.append(bin_seg_pred)
            self.targets_list_train.append(original_targets)
            self.datasets_list_train += batch[-1]
        if self.one_leg_model:
            if self.uaw_loss:
                loss, mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, reg_l1_loss, mse_bin_loss, mse_loss_uaw, mse_bin_loss_uaw, bce_loss_uaw, mhd_loss_gt_uaw, mhd_loss_pred_uaw = self.calc_loss_one_leg_with_uaw(
                    sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets)
                self.log('scaled_train_mse_loss', mse_loss_uaw, prog_bar=False, on_epoch=True, on_step=False)
                self.log('scaled_train_mse_bin_loss', mse_bin_loss_uaw, prog_bar=False, on_epoch=True, on_step=False)
                self.log('scaled_train_bce_loss', bce_loss_uaw, prog_bar=False, on_epoch=True, on_step=False)
                self.log('scaled_train_mhd_loss SD_pred*(gt==0)', mhd_loss_gt_uaw, prog_bar=False, on_epoch=True,
                         on_step=False)
                self.log('scaled_train_mhd_loss (pred==0)*SD_GT', mhd_loss_pred_uaw, prog_bar=False, on_epoch=True,
                         on_step=False)
            else:
                # loss, mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, l1_loss, mse_bin_loss = self.calc_loss_one_leg(
                #     sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets)
                loss, mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, reg_l1_loss, l1_loss, huber_loss, focal_loss = self.calc_loss_one_leg2(
                    sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets)
            self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
            self.log('train_loss_with_mhd_pred', loss + mhd_loss_pred, prog_bar=True, on_epoch=True, on_step=False)

            self.log('train_mse_loss', mse_loss, prog_bar=False, on_epoch=True, on_step=False)
            self.log('train_l1_loss', l1_loss, prog_bar=False, on_epoch=True, on_step=False)
            self.log('train_huber_loss', huber_loss, prog_bar=False, on_epoch=True, on_step=False)
            self.log('train_bce_loss', bce_loss, prog_bar=False, on_epoch=True, on_step=False)
            self.log('train_focal_loss', focal_loss, prog_bar=False, on_epoch=True, on_step=False)
            self.log('train_mhd_loss SD_pred*(gt==0)', mhd_loss_gt, prog_bar=False, on_epoch=True, on_step=False)
            self.log('train_mhd_loss (pred==0)*SD_GT', mhd_loss_pred, prog_bar=False, on_epoch=True, on_step=False)
            self.log('train__reg_l1_loss', reg_l1_loss, prog_bar=False, on_epoch=True, on_step=False)
            return loss
        loss, mse_loss, bce_loss, bce_loss_of_sign_dist, mhd_loss_gt, mhd_loss_pred, l1_loss = self.calc_loss(
            sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

        self.log('train_mse_loss', mse_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_bce_loss', bce_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_bce_loss_of_sign_dist', bce_loss_of_sign_dist, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_mhd_loss SD_pred*(gt==0)', mhd_loss_gt, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_mhd_loss (pred==0)*SD_GT', mhd_loss_pred, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_l1_loss', l1_loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 5 == 0:
            print(f"\nCalculating train Jaccard Index for epoch: {self.current_epoch}")

            seg_scores, det_score = check_accuracy(self.preds_signDist_list_train, self.preds_bin_list_train,
                                        self.targets_list_train,
                                        batch_size=self.batch_size, three_d=False, _2_5d=False)

            self.log("train_SEG", seg_scores.mean(), on_epoch=True, prog_bar=True)
            print(f"total train SEG: mean={seg_scores.mean():.3f}, std={seg_scores.std():.3f}")
            scores_by_dataset = self.get_scores_by_dataset(self.datasets_list_train, seg_scores)
            for ds, scores in scores_by_dataset.items():
                scores = np.array(scores)
                mean = scores.mean()
                std = scores.std()
                print(f"{ds}: mean={mean:.3f}, std={std:.3f}")
                self.log(f"train_SEG_{ds}", mean, on_epoch=True, prog_bar=True)
            self.preds_signDist_list_train.clear()
            self.preds_bin_list_train.clear()
            self.targets_list_train.clear()
            self.datasets_list_train.clear()

    def validation_step(self, batch, batch_idx):
        sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets, original_targets = self.predict_batch(batch)
        if (self.current_epoch + 1) % 5 == 0:
            self.preds_signDist_list.append(sign_dist_pred)
            self.preds_bin_list.append(bin_seg_pred)
            self.targets_list.append(original_targets)
            self.datasets_list += batch[-1]
            if not self.test_sign_dist_targets:
                self.test_sign_dist_targets.append(sign_dist_targets)
                self.data_list.append(batch[0])
        if self.one_leg_model:
            if self.uaw_loss:
                loss, mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, reg_l1_loss, mse_bin_loss, mse_loss_uaw, mse_bin_loss_uaw, bce_loss_uaw, mhd_loss_gt_uaw, mhd_loss_pred_uaw = self.calc_loss_one_leg_with_uaw(
                    sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets)
                self.log('scaled_val_mse_loss', mse_loss_uaw, prog_bar=False, on_epoch=True, on_step=False)
                self.log('scaled_val_mse_bin_loss', mse_bin_loss_uaw, prog_bar=False, on_epoch=True, on_step=False)
                self.log('scaled_val_bce_loss', bce_loss_uaw, prog_bar=False, on_epoch=True, on_step=False)
                self.log('scaled_val_mhd_loss SD_pred*(gt==0)', mhd_loss_gt_uaw, prog_bar=False, on_epoch=True,
                         on_step=False)
                self.log('scaled_val_mhd_loss (pred==0)*SD_GT', mhd_loss_pred_uaw, prog_bar=False, on_epoch=True,
                         on_step=False)
            else:
                # loss, mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, l1_loss, mse_bin_loss = self.calc_loss_one_leg(
                #     sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets)
                loss, mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, reg_l1_loss, l1_loss, huber_loss, focal_loss = self.calc_loss_one_leg2(
                    sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets)
            self.log("val_loss", loss, batch_size=batch[0].shape[0], on_epoch=True, on_step=False)
            self.log('val_loss_with_mhd_pred', loss + mhd_loss_pred, prog_bar=True, on_epoch=True, on_step=False)

            self.log('val_mse_loss', mse_loss, prog_bar=False, on_epoch=True, on_step=False)
            self.log('val_huber_loss', huber_loss, prog_bar=False, on_epoch=True, on_step=False)
            self.log('val_l1_loss', l1_loss, prog_bar=False, on_epoch=True, on_step=False)
            self.log('val_bce_loss', bce_loss, prog_bar=False, on_epoch=True, on_step=False)
            self.log('val_focal_loss', focal_loss, prog_bar=False, on_epoch=True, on_step=False)
            self.log('val_mhd_loss SD_pred*(gt==0)', mhd_loss_gt, prog_bar=False, on_epoch=True, on_step=False)
            self.log('val_mhd_loss (pred==0)*SD_GT', mhd_loss_pred, prog_bar=False, on_epoch=True, on_step=False)
            self.log('val_reg_l1_loss', reg_l1_loss, prog_bar=False, on_epoch=True, on_step=False)
            return loss

        loss, mse_loss, bce_loss, bce_loss_of_sign_dist, mhd_loss_gt, mhd_loss_pred, l1_loss = self.calc_loss(
            sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets)

        self.log("val_loss", loss, batch_size=batch[0].shape[0], on_epoch=True, on_step=False)

        self.log('val_mse_loss', mse_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_bce_loss', bce_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_bce_loss_of_sign_dist', bce_loss_of_sign_dist, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_mhd_loss SD_pred*(gt==0)', mhd_loss_gt, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_mhd_loss (pred==0)*SD_GT', mhd_loss_pred, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_l1_loss', l1_loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_end(self):
        def plot_image(image, cmap, title, colorbar, title_image=None):
            fig, axs = plt.subplots()
            im = axs.imshow(image, cmap=cmap)
            axs.axis("off")
            if colorbar:
                plt.colorbar(im, ax=axs, orientation='vertical', fraction=0.046, pad=0.04)
            fig.tight_layout()
            if title_image is None:
                self.trainer.logger.experiment.log({title: wandb.Image(fig)})
            else:
                axs.set_title(title)
                self.trainer.logger.experiment.log({title_image: wandb.Image(fig)})
            plt.close(fig)

        if self.one_leg_model:
            self.log("alpha", self.model.alpha.item(), prog_bar=False)
            self.log("beta", self.model.beta.item(), prog_bar=False)
            if self.uaw_loss:
                self.log("log_sigma2_mse", self.model.log_sigma2_mse.item(), prog_bar=False)
                self.log("log_sigma2_mse_bin", self.model.log_sigma2_mse_bin.item(), prog_bar=False)
                self.log("log_sigma2_bce", self.model.log_sigma2_bce.item(), prog_bar=False)
                self.log("log_sigma2_mhd_gt", self.model.log_sigma2_mhd_gt.item(), prog_bar=False)
                self.log("log_sigma2_mhd_pred", self.model.log_sigma2_mhd_pred.item(), prog_bar=False)
        if (self.current_epoch + 1) % 5 == 0:
            print(f"\nCalculating val Jaccard Index for epoch: {self.current_epoch}")

            seg_scores, det_scores = check_accuracy(self.preds_signDist_list, self.preds_bin_list, self.targets_list,
                                        batch_size=self.batch_size, three_d=False, _2_5d=False)
            self.log("val_SEG", seg_scores.mean(), on_epoch=True, prog_bar=True)
            print(f"total val SEG: mean={seg_scores.mean():.3f}, std={seg_scores.std():.3f}")
            scores_by_dataset = self.get_scores_by_dataset(self.datasets_list, seg_scores)
            for ds, scores in scores_by_dataset.items():
                scores = np.array(scores)
                mean = scores.mean()
                std = scores.std()
                print(f"{ds}: mean={mean:.3f}, std={std:.3f}")
                self.log(f"val_SEG_{ds}", mean, on_epoch=True, prog_bar=True)
            input_image = self.data_list[0][0, 0].cpu().numpy()
            sd_image = self.preds_signDist_list[0][0, 0].cpu().numpy()
            gt_image = self.targets_list[0][0, 0].cpu().numpy()

            bin_pred_hard_0_5 = (torch.sigmoid(self.preds_bin_list[0]) > 0.5)[0, 0].cpu().numpy()
            bin_pred_soft = torch.sigmoid(self.preds_bin_list[0])[0, 0].cpu().numpy()
            inv_bin_pred_soft = 1 - bin_pred_soft

            if not self.one_leg_model:
                edge_map = self.detect_edges_grad(self.preds_signDist_list[0])
            elif self.kernel_type is None:
                edge_map = self.detect_edges_grad(self.preds_bin_list[0])
            elif self.kernel_type == 'sigmoid':
                edge_map = self.detect_edges_sigmoid(self.preds_bin_list[0])
            else:
                edge_map = self.detect_edges_laplac(self.preds_bin_list[0],  # bin_pred_hard_0_7,
                                                    kernel_type=self.kernel_type)
            edge_map_np = edge_map[0, 0].cpu().numpy()
            #     np.save(f"/home/thomasm/workspace/cells_lightning/cells/edge_map_test{self.current_epoch}.npy", edge_map)
            #     np.save(f"/home/thomasm/workspace/cells_lightning/cells/bin_pred_soft{self.current_epoch}.npy", bin_pred_soft)

            # semantic_pred = predicted_classes[0]*bin_pred
            # instance_pred, _ = get_cell_instances(semantic_pred, three_d=False)
            instance_pred, _ = get_cell_instances(bin_pred_hard_0_5, three_d=False)
            instance_pred[instance_pred != 0] += 5

            fig, axs = plt.subplots(2, 5, figsize=(15, 5))
            axs[0, 0].imshow(input_image, cmap='gray')
            axs[0, 0].set_title("Input Image")
            axs[0, 0].axis('off')

            im0 = axs[0, 1].imshow(sd_image, cmap='jet')
            axs[0, 1].set_title("SD Prediction")
            axs[0, 1].axis('off')
            plt.colorbar(im0, ax=axs[0, 1], orientation='vertical', fraction=0.046, pad=0.04)

            im1 = axs[0, 2].imshow(bin_pred_soft, cmap='jet')
            axs[0, 2].set_title("Soft Binary Prediction")
            axs[0, 2].axis('off')
            plt.colorbar(im1, ax=axs[0, 2], orientation='vertical', fraction=0.046, pad=0.04)

            axs[0, 3].imshow(bin_pred_hard_0_5, cmap='viridis')
            axs[0, 3].set_title(f"Binary Prediction > 0.5")
            axs[0, 3].axis('off')

            axs[0, 4].imshow(instance_pred, cmap='nipy_spectral')
            axs[0, 4].set_title(f"Instance Prediction")
            axs[0, 4].axis('off')

            im1 = axs[1, 0].imshow(inv_bin_pred_soft, cmap='jet')
            axs[1, 0].set_title("Inverse Soft Binary Prediction")
            axs[1, 0].axis('off')
            plt.colorbar(im1, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)

            im2 = axs[1, 1].imshow(edge_map_np, cmap='jet')
            axs[1, 1].set_title("Prediction Edges Map")
            axs[1, 1].axis('off')
            plt.colorbar(im2, ax=axs[1, 1], orientation='vertical', fraction=0.046, pad=0.04)

            im3 = axs[1, 2].imshow(self.test_sign_dist_targets[0][0, 0].cpu().numpy(), cmap='jet')
            axs[1, 2].set_title("SD Ground Truth")
            axs[1, 2].axis('off')
            plt.colorbar(im3, ax=axs[1, 2], orientation='vertical', fraction=0.046, pad=0.04)

            axs[1, 3].imshow(gt_image > 0, cmap='viridis')
            axs[1, 3].set_title("Semantic Ground Truth")
            axs[1, 3].axis('off')

            axs[1, 4].imshow(gt_image, cmap='nipy_spectral')
            axs[1, 4].set_title("Ground Truth")
            axs[1, 4].axis('off')

            plt.tight_layout()

            self.trainer.logger.experiment.log({"image": wandb.Image(fig)})
            plt.close(fig)

            # self.preds_bin_list[0]
            # self.test_sign_dist_targets[0]
            edge_mask_gt = self.detect_edges_sigmoid(self.test_sign_dist_targets[0])
            tanh_pred = torch.tanh(self.preds_bin_list[0])
            lmhd = (torch.abs(tanh_pred) * edge_mask_gt) / float(torch.sum(edge_mask_gt).item())

            edge_mask_pred = self.detect_edges_sigmoid(self.preds_bin_list[0])
            tanh_gt = torch.tanh(self.test_sign_dist_targets[0])
            rmhd = (torch.abs(tanh_gt) * edge_mask_pred) / float(torch.sum(edge_mask_pred).item())

            #mhd loss plot
            fig, axs = plt.subplots(2, 4, figsize=(15, 5))
            im = axs[0, 0].imshow(edge_mask_gt[0, 0].cpu().numpy(), cmap='jet')
            axs[0, 0].set_title("Edge Map GT")
            axs[0, 0].axis('off')
            plt.colorbar(im, ax=axs[0, 0], orientation='vertical', fraction=0.046, pad=0.04)

            im = axs[0, 1].imshow(tanh_pred[0, 0].cpu().numpy(), cmap='jet')
            axs[0, 1].set_title(r"$\tanh_{\alpha,\beta}(\phi_{PRED})$")
            axs[0, 1].axis('off')
            plt.colorbar(im, ax=axs[0, 1], orientation='vertical', fraction=0.046, pad=0.04)

            im = axs[0, 2].imshow(edge_mask_pred[0, 0].cpu().numpy(), cmap='jet')
            axs[0, 2].set_title("Edge Map Pred")
            axs[0, 2].axis('off')
            plt.colorbar(im, ax=axs[0, 2], orientation='vertical', fraction=0.046, pad=0.04)

            im = axs[0, 3].imshow(tanh_gt[0, 0].cpu().numpy(), cmap='jet')
            axs[0, 3].set_title(r"$\tanh_{\alpha,\beta}(\phi_{GT})$")
            axs[0, 3].axis('off')
            plt.colorbar(im, ax=axs[0, 3], orientation='vertical', fraction=0.046, pad=0.04)

            im = axs[1, 0].imshow(lmhd[0, 0].cpu().numpy(), cmap='jet')
            axs[1, 0].set_title(f"LMHD={torch.sum(lmhd[0, 0]):.3f}, alpha={self.model.alpha.item():.3f} , beta={self.model.beta.item():.3f}")
            axs[1, 0].axis('off')
            plt.colorbar(im, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)

            im = axs[1, 2].imshow(rmhd[0, 0].cpu().numpy(), cmap='jet')
            axs[1, 2].set_title(f"RMHD={torch.sum(rmhd[0, 0]):.3f}, alpha={self.model.alpha.item():.3f}, beta={self.model.beta.item():.3f}")
            axs[1, 2].axis('off')
            plt.colorbar(im, ax=axs[1, 2], orientation='vertical', fraction=0.046, pad=0.04)
            plt.tight_layout()

            self.trainer.logger.experiment.log({"mhd loss plot": wandb.Image(fig)})
            plt.close(fig)

            # plot_image(image, cmap, title, colorbar)
            plot_image(image=input_image, cmap='gray', title="Input Image", colorbar=False)
            plot_image(image=sd_image, cmap='jet', title="SD Prediction", colorbar=True)
            plot_image(image=bin_pred_soft, cmap='jet', title="Soft Binary Prediction", colorbar=True)
            plot_image(image=instance_pred, cmap='nipy_spectral', title="Instance Prediction", colorbar=False)
            plot_image(image=edge_map_np, cmap='jet', title="Prediction Edges Map", colorbar=True)
            plot_image(image=self.test_sign_dist_targets[0][0, 0].cpu().numpy(), cmap='jet', title="SD Ground Truth", colorbar=True)
            plot_image(image=gt_image, cmap='nipy_spectral', title="Ground Truth", colorbar=True)
            plot_image(image=edge_mask_gt[0, 0].cpu().numpy(), cmap='jet', title="GT Edge Map", colorbar=True)
            plot_image(image=tanh_pred[0, 0].cpu().numpy(), cmap='jet', title=r"$\tanh_{\alpha,\beta}(\phi_{PRED})$", colorbar=True)
            plot_image(image=tanh_gt[0, 0].cpu().numpy(), cmap='jet', title=r"$\tanh_{\alpha,\beta}(\phi_{GT})$", colorbar=True)
            plot_image(image=lmhd[0, 0].cpu().numpy(), cmap='jet', title=f"LMHD={torch.sum(lmhd[0, 0]):.3f}", title_image="LMHD", colorbar=True)
            plot_image(image=rmhd[0, 0].cpu().numpy(), cmap='jet', title=f"RMHD={torch.sum(rmhd[0, 0]):.3f}", title_image="RMHD", colorbar=True)
            # plot_image(image=.cpu().numpy(), cmap='jet', title=f"MSE={torch.sum():.3f}", title_image="MSE", colorbar=True)
            # plot_image(image=.cpu().numpy(), cmap = 'jet', title = f"L1={torch.sum():.3f}", title_image = "L1", colorbar = True)
            # plot_image(image=.cpu().numpy(), cmap = 'jet', title = f"BCE={torch.sum():.3f}", title_image = "BCE", colorbar = True)





            # sim_sign_dist_pred, sim_bin_seg_pred = self.model(
            #     torch.from_numpy(
            #         (self.image_test_sim - self.image_test_sim.mean()) / (self.image_test_sim.std())).unsqueeze(
            #         0).unsqueeze(0).to(self.preds_bin_list[0].device))
            # hela_sign_dist_pred, hela_bin_seg_pred = self.model(
            #     torch.from_numpy(
            #         (self.image_test_hela - self.image_test_hela.mean()) / (self.image_test_hela.std())).unsqueeze(
            #         0).unsqueeze(0).to(self.preds_bin_list[0].device))
            #
            # sim_gt_sign_dist = self.signed_distance_function(
            #     torch.from_numpy(self.mask_test_sim).unsqueeze(0).to(self.preds_bin_list[0].device)).unsqueeze(0)
            # hela_gt_sign_dist = self.signed_distance_function(
            #     torch.from_numpy(self.mask_test_hela).unsqueeze(0).to(self.preds_bin_list[0].device)).unsqueeze(0)
            #
            # if not self.one_leg_model:
            #     sim_edge_map = self.detect_edges_grad(sim_sign_dist_pred)
            #     hela_edge_map = self.detect_edges_grad(hela_sign_dist_pred)
            # elif self.kernel_type is None:
            #     sim_edge_map = self.detect_edges_grad(sim_bin_seg_pred)
            #     hela_edge_map = self.detect_edges_grad(hela_bin_seg_pred)
            # elif self.kernel_type == 'sigmoid':
            #     sim_edge_map = self.detect_edges_sigmoid(sim_bin_seg_pred)
            #     hela_edge_map = self.detect_edges_sigmoid(hela_bin_seg_pred)
            # else:
            #     sim_edge_map = self.detect_edges_laplac(sim_bin_seg_pred,
            #                                             kernel_type=self.kernel_type)
            #     hela_edge_map = self.detect_edges_laplac(hela_bin_seg_pred,
            #                                              kernel_type=self.kernel_type)
            # sim_edge_map_np = sim_edge_map[0, 0].cpu().numpy()
            # hela_edge_map_np = hela_edge_map[0, 0].cpu().numpy()
            #
            # sim_pred_edges_on_image = self.overlay_edges_on_grayscale(self.image_test_sim,
            #                                                           sim_edge_map_np,
            #                                                           (sim_gt_sign_dist == 0)[0, 0].cpu().numpy())
            #
            # sim_mhd_loss_gt = self.modified_hausdorff_distance(sim_bin_seg_pred,
            #                                                    (sim_gt_sign_dist == 0))
            # sim_mhd_loss_pred = self.modified_hausdorff_distance(sim_gt_sign_dist, sim_edge_map)
            #
            # seg_scores_sim, det_scores_sim = check_accuracy([sim_sign_dist_pred], [sim_bin_seg_pred],
            #                                 [torch.from_numpy(self.mask_test_sim).unsqueeze(0).unsqueeze(0).to(
            #                                     self.preds_bin_list[0].device)],
            #                                 batch_size=1, three_d=False, _2_5d=False)
            # seg_scores_hela, det_scores_hela = check_accuracy([hela_sign_dist_pred], [hela_bin_seg_pred],
            #                                  [torch.from_numpy(self.mask_test_hela).unsqueeze(0).unsqueeze(0).to(
            #                                      self.preds_bin_list[0].device)],
            #                                  batch_size=1, three_d=False, _2_5d=False)
            #
            # fig, axs = plt.subplots()
            # axs.imshow(sim_pred_edges_on_image)
            # axs.axis("off")
            # axs.set_title(
            #     f"mhd_pred: {sim_mhd_loss_pred:.2f} | mhd_gt: {sim_mhd_loss_gt:.2f} | SEG: {seg_scores_sim.mean():.2f}")
            # self.logger.experiment.log(
            #     {f"sim+ (red-prediction, green-gt)": wandb.Image(fig)})
            # plt.close(fig)
            #
            # hela_pred_edges_on_image = self.overlay_edges_on_grayscale(self.image_test_hela,
            #                                                            hela_edge_map_np,
            #                                                            (hela_gt_sign_dist == 0)[0, 0].cpu().numpy())
            #
            # hela_mhd_loss_gt = self.modified_hausdorff_distance(hela_bin_seg_pred,
            #                                                     (hela_gt_sign_dist == 0))
            # hela_mhd_loss_pred = self.modified_hausdorff_distance(hela_gt_sign_dist, hela_edge_map)
            #
            # fig, axs = plt.subplots()
            # axs.imshow(hela_pred_edges_on_image)
            # axs.axis("off")
            # axs.set_title(
            #     f"mhd_pred: {hela_mhd_loss_pred:.2f} | mhd_gt: {hela_mhd_loss_gt:.2f} | SEG: {seg_scores_hela.mean():.2f}")
            # self.logger.experiment.log(
            #     {f"HeLa (red-prediction, green-gt)": wandb.Image(fig)})
            # plt.close(fig)

            self.preds_signDist_list.clear()
            self.preds_bin_list.clear()
            self.targets_list.clear()
            self.test_sign_dist_targets.clear()
            self.data_list.clear()
            self.datasets_list.clear()

    def test_step(self, batch, batch_idx):
        data, _, dataset, mask_name = batch

        sign_dist_pred, bin_seg_pred = self.model(data)

        self.preds_signDist_list.append(sign_dist_pred)
        self.preds_bin_list.append(bin_seg_pred)
        self.data_list.append(data)
        self.datasets_list += dataset
        self.mask_name_list += mask_name

    def on_test_epoch_end(self):
        if self.output_dir is None:
            save_root = Path("/raid/data/users/thomasm/pred_mask2")
            # save_root =  "/gpfs0/tamyr/users/thomasm/workspace/"
        else:
            save_root = Path(self.output_dir)

        for bin_pred, dataset, mask_name in zip(self.preds_bin_list, self.datasets_list, self.mask_name_list):
            bin_pred_hard_0_5 = (torch.sigmoid(bin_pred) > 0.5)[0, 0].cpu().numpy()
            instance_pred, _ = get_cell_instances(bin_pred_hard_0_5, three_d=False)
            out_path = save_root / mask_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tiff.imwrite(out_path, instance_pred.astype(np.uint16))

    # def test_step(self, batch, batch_idx):
    #     sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets, original_targets = self.predict_batch(batch)
    #     # torch.save(bin_seg_pred, "/gpfs0/tamyr/users/thomasm/workspace/ckpts/bin_seg_pred_for_tests.pt")
    #     # exit()
    #     self.preds_signDist_list.append(sign_dist_pred)
    #     self.preds_bin_list.append(bin_seg_pred)
    #     self.targets_list.append(original_targets)
    #     self.test_sign_dist_targets.append(sign_dist_targets)
    #     self.data_list.append(batch[0])
    #     self.datasets_list += batch[-1]
    #
    # def on_test_epoch_end(self):
    #     def overlay_edges_on_sdf(sd, edge_mask):
    #         overlay_tensor = sd.clone()  # Clone SDF tensor to avoid modifying the original
    #         overlay_tensor[edge_mask == 1] = sd.max() + 2
    #         return overlay_tensor[0, 0].cpu().numpy()
    #
    #     print(f"\nCalculating Jaccard Index")
    #
    #     seg_scores, det_scores = check_accuracy(self.preds_signDist_list, self.preds_bin_list, self.targets_list,
    #                                 batch_size=self.batch_size, three_d=False, _2_5d=False)
    #
    #     self.log("test_SEG", seg_scores.mean(), on_epoch=True, prog_bar=True)
    #     print(f"total test SEG: mean={seg_scores.mean():.3f}, std={seg_scores.std():.3f}")
    #     print(f"total test DEG: mean={det_scores.mean():.3f}, std={det_scores.std():.3f}")
    #     seg_scores_by_dataset = self.get_scores_by_dataset(self.datasets_list, seg_scores)
    #     det_scores_by_dataset = self.get_scores_by_dataset(self.datasets_list, det_scores)
    #     print("SEG:")
    #     for ds, scores in seg_scores_by_dataset.items():
    #         scores = np.array(scores)
    #         mean = scores.mean()
    #         std = scores.std()
    #         print(f"\t{ds}: mean={mean:.3f}, std={std:.3f}, len: {len(scores)}")
    #         self.log(f"test_SEG_{ds}", mean, on_epoch=True, prog_bar=True)
    #     print("DET:")
    #     for ds, scores in det_scores_by_dataset.items():
    #         scores = np.array(scores)
    #         mean = scores.mean()
    #         std = scores.std()
    #         print(f"\t{ds}: mean={mean:.3f}, std={std:.3f}, len: {len(scores)}")
    #         self.log(f"test_DEG_{ds}", mean, on_epoch=True, prog_bar=True)


        ###############################################################################
        pred_labels = []
        # for bin_seg_pred in self.preds_bin_list:
        #     bin_seg_pred = (torch.sigmoid(bin_seg_pred) > 0.5).squeeze(0).squeeze(0).cpu().numpy()
        #     pred_labels_mask, _ = get_cell_instances(bin_seg_pred, three_d=False)
        #     pred_labels.append(pred_labels_mask)
        #
        # volume_pred = self.stack_2d_labels_to_3d(pred_labels)
        # # volume_pred = torch.cat(self.preds_bin_list, dim=0).squeeze(1)
        # # volume_pred = (torch.sigmoid(volume_pred) > 0.5).cpu().numpy()
        # # volume_pred, _ = get_cell_instances(volume_pred, three_d=True)
        # volume_gt = torch.cat(self.targets_list, dim=0) # .squeeze(1)
        # volume_gt = volume_gt.squeeze(1).cpu().numpy().astype(np.uint8)
        # volume_data = torch.cat(self.data_list, dim=0).squeeze(1)
        #
        # tiff.imwrite("/home/thomasm/workspace/cells_lightning/cells/images/NT_240423_9hr_data.tif", volume_data.cpu().numpy())
        # tiff.imwrite("/home/thomasm/workspace/cells_lightning/cells/images/NT_240423_9hr_pred2.tif", volume_pred.astype(np.uint8))
        # tiff.imwrite("/home/thomasm/workspace/cells_lightning/cells/images/NT_240423_9hr_gt.tif", volume_gt.astype(np.uint8))

        #
        # print("before img2")
        # print(f"volume_pred.shape: {volume_pred.shape}")
        # # self.log_3d_scatter_to_wandb(volume_pred.astype(np.uint8), name="3D_pred")
        # self.log_3d_mesh_to_wandb(volume_pred.astype(np.uint8), name="3D_pred")
        # print("before img3")
        # print(f"volume_gt.shape: {volume_gt.shape}")
        # # self.log_3d_scatter_to_wandb(volume_gt, name="3D_gt")
        # self.log_3d_mesh_to_wandb(volume_gt, name="3D_gt")
        #
        # for i in range(len(self.preds_bin_list)):
        #     input_image = self.data_list[i][0, 0].cpu().numpy()
        #
        #     semantic_pred = (torch.sigmoid(self.preds_bin_list[i]) > 0.5)[0, 0].cpu().numpy()
        #     instance_pred, _ = get_cell_instances(semantic_pred, three_d=False)
        #     instance_pred[instance_pred != 0] += 5
        #
        #     gt_image = self.targets_list[i][0, 0].cpu().numpy()
        #     edge_signDist_pred = self.detect_edges_sigmoid(self.preds_bin_list[i])
        #     pred_edges_on_image = self.overlay_edges_on_grayscale(input_image,
        #                                                       edge_signDist_pred[0, 0].cpu().numpy(),
        #                                                       (self.test_sign_dist_targets[i] == 0)[
        #                                                           0, 0].cpu().numpy())
        #
        #     seg_scores, det_scores = check_accuracy(self.preds_signDist_list[i], self.preds_bin_list[i],
        #                                             self.targets_list[i], batch_size=1,
        #                                             three_d=False, _2_5d=False)
        #     fig, axs = plt.subplots()
        #     axs.imshow(pred_edges_on_image)
        #     axs.axis("off")
        #     axs.set_title(
        #         f"SEG: {seg_scores.mean():.2f}")
        #     self.logger.experiment.log(
        #         {f"image with edge of prediction(red-prediction, green-gt)": wandb.Image(fig)})
        #     plt.close(fig)
        #
        #     fig, axs = plt.subplots(2, 2)
        #     axs[0, 0].imshow(input_image, cmap='gray')
        #     axs[0, 0].set_title("Input Image")
        #     axs[0, 0].axis('off')
        #
        #     axs[0, 1].imshow(instance_pred, cmap='nipy_spectral')
        #     axs[0, 1].set_title(f"Instance Prediction")
        #     axs[0, 1].axis('off')
        #
        #     axs[1, 0].imshow(pred_edges_on_image)
        #     axs[1, 0].set_title(f"Overlay: GT (Green), Pred (Red)")
        #     axs[1, 0].axis('off')
        #
        #     axs[1, 1].imshow(gt_image, cmap='nipy_spectral')
        #     axs[1, 1].set_title("Ground Truth")
        #     axs[1, 1].axis('off')
        #     self.trainer.logger.experiment.log({"for presentation": wandb.Image(fig)})
        #     plt.close(fig)
            ################################################################### the changes ended here

        # total_mse_loss = 0
        # total_mhd_loss_gt = 0
        # total_mhd_loss_pred = 0
        # for i in range(len(self.preds_signDist_list)):
        #     mse_loss_fn = nn.MSELoss()
        #     mse_loss = mse_loss_fn(self.preds_signDist_list[i], self.test_sign_dist_targets[i])
        #     total_mse_loss += mse_loss.item()
        #
        #     mhd_loss_gt = self.modified_hausdorff_distance(self.preds_signDist_list[i],
        #                                                    (self.test_sign_dist_targets[i] == 0))
        #     total_mhd_loss_gt += mhd_loss_gt
        #
        #     # edge_signDist_pred = self.detect_edges_grad(self.preds_signDist_list[i])
        #     edge_signDist_pred = self.detect_edges_sigmoid(self.preds_bin_list[i])
        #     mhd_loss_pred = self.modified_hausdorff_distance(self.test_sign_dist_targets[i], edge_signDist_pred)
        #     total_mhd_loss_pred += mhd_loss_pred
        #
        #     # if i < 10 or (i >= (len(self.preds_signDist_list) - 19) and i <= (len(self.preds_signDist_list) - 10)):
        #     if i < 50 or (i >= (len(self.preds_signDist_list) - 69) and i <= (len(self.preds_signDist_list) - 10)):
        #         input_image = self.data_list[i][0, 0].cpu().numpy()
        #         sd_image = self.preds_signDist_list[i][0, 0].cpu().numpy()
        #         gt_image = self.targets_list[i][0, 0].cpu().numpy()
        #         bin_pred = torch.sigmoid(self.preds_bin_list[i])[0, 0].cpu().numpy()
        #         semantic_pred = (torch.sigmoid(self.preds_bin_list[i]) > 0.5)[0, 0].cpu().numpy()
        #         instance_pred, _ = get_cell_instances(semantic_pred, three_d=False)
        #         instance_pred[instance_pred != 0] += 5
        #
        #         # self.log('val_mhd_loss SD_pred*(gt==0)', float(f"{mhd_loss_gt:.3f}"))
        #         # self.log('val_mhd_loss (pred==0)*SD_GT', float(f"{mhd_loss_pred:.3f}"))
        #
        #         wandb.log({'test_mhd_loss (pred==0)*SD_GT': float(f"{mhd_loss_pred:.3f}")})
        #         wandb.log({'test_mhd_loss SD_pred*(gt==0)': float(f"{mhd_loss_gt:.3f}")})
        #
        #         fig, axs = plt.subplots(3, 5, figsize=(15, 5))
        #         axs[0, 0].imshow(input_image, cmap='gray')
        #         axs[0, 0].set_title("Input Image")
        #         axs[0, 0].axis('off')
        #
        #         im0 = axs[0, 1].imshow(sd_image, cmap='jet')
        #         axs[0, 1].set_title("SD Prediction")
        #         axs[0, 1].axis('off')
        #         plt.colorbar(im0, ax=axs[0, 1], orientation='vertical', fraction=0.046, pad=0.04)
        #
        #         im1 = axs[0, 2].imshow(bin_pred, cmap='jet')
        #         axs[0, 2].set_title("Soft Binary Prediction")
        #         axs[0, 2].axis('off')
        #         plt.colorbar(im1, ax=axs[0, 2], orientation='vertical', fraction=0.046, pad=0.04)
        #
        #         axs[0, 3].imshow(semantic_pred, cmap='viridis')
        #         axs[0, 3].set_title(f"Binary Prediction > 0.5")
        #         axs[0, 3].axis('off')
        #
        #         axs[0, 4].imshow(instance_pred, cmap='nipy_spectral')
        #         axs[0, 4].set_title(f"Instance Prediction")
        #         axs[0, 4].axis('off')
        #
        #         im1 = axs[1, 0].imshow(1 - bin_pred, cmap='jet')
        #         axs[1, 0].set_title("Inverse Soft Binary Prediction")
        #         axs[1, 0].axis('off')
        #         plt.colorbar(im1, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
        #
        #         im2 = axs[1, 1].imshow(edge_signDist_pred[0, 0].cpu().numpy(), cmap='jet')
        #         axs[1, 1].set_title("Prediction Edges Map")
        #         axs[1, 1].axis('off')
        #         plt.colorbar(im2, ax=axs[1, 1], orientation='vertical', fraction=0.046, pad=0.04)
        #
        #         axs[1, 2].imshow(self.test_sign_dist_targets[i][0, 0].cpu().numpy(), cmap='jet')
        #         axs[1, 2].set_title("SD Ground Truth")
        #         axs[1, 2].axis('off')
        #
        #         axs[1, 3].imshow(gt_image > 0, cmap='viridis')
        #         axs[1, 3].set_title("Semantic Ground Truth")
        #         axs[1, 3].axis('off')
        #
        #         axs[1, 4].imshow(gt_image, cmap='nipy_spectral')
        #         axs[1, 4].set_title("Ground Truth")
        #         axs[1, 4].axis('off')
        #
        #         axs[2, 0].imshow((self.test_sign_dist_targets[i][0, 0] == 0).cpu().numpy(), cmap='jet')
        #         axs[2, 0].set_title("Ground Truth Edges Map")
        #         axs[2, 0].axis('off')
        #         plt.tight_layout()
        #
        #         self.trainer.logger.experiment.log({"image": wandb.Image(fig)})
        #         plt.close(fig)
        #
        #         pred_sd_gt_edges = overlay_edges_on_sdf(self.preds_signDist_list[i],
        #                                                 self.test_sign_dist_targets[i] == 0)
        #         pred_edges_gt_sd = overlay_edges_on_sdf(self.test_sign_dist_targets[i], edge_signDist_pred >= 0.85)
        #
        #         fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        #         im0 = axs[0].imshow(pred_sd_gt_edges, cmap='jet')
        #         axs[0].set_title(f"mhd_loss SD_pred*(gt==0) = {mhd_loss_gt:.3f}")
        #         axs[0].axis('off')
        #         plt.colorbar(im0, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
        #
        #         im1 = axs[1].imshow(pred_edges_gt_sd, cmap='jet')
        #         axs[1].set_title(f"mhd_loss (pred==0)*SD_GT = {mhd_loss_pred:.3f}")
        #         axs[1].axis('off')
        #         plt.colorbar(im1, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        #
        #         plt.tight_layout()
        #
        #         self.trainer.logger.experiment.log({"image_mhd": wandb.Image(fig)})
        #         plt.close(fig)
        #
        #         pred_edges_on_image = self.overlay_edges_on_grayscale(input_image,
        #                                                               edge_signDist_pred[0, 0].cpu().numpy(),
        #                                                               (self.test_sign_dist_targets[i] == 0)[
        #                                                                   0, 0].cpu().numpy())
        #
        #         seg_scores, det_scores = check_accuracy(self.preds_signDist_list[i], self.preds_bin_list[i],
        #                                     self.targets_list[i], batch_size=1,
        #                                     three_d=False, _2_5d=False)
        #         fig, axs = plt.subplots()
        #         axs.imshow(pred_edges_on_image)
        #         axs.axis("off")
        #         axs.set_title(
        #             f"mhd_pred: {mhd_loss_pred:.2f} | mhd_gt: {mhd_loss_gt:.2f} | SEG: {seg_scores.mean():.2f}")
        #         self.logger.experiment.log(
        #             {f"image with edge of prediction(red-prediction, green-gt)": wandb.Image(fig)})
        #         plt.close(fig)
        #
        #         fig, axs = plt.subplots(2, 2)
        #         axs[0, 0].imshow(input_image, cmap='gray')
        #         axs[0, 0].set_title("Input Image")
        #         axs[0, 0].axis('off')
        #
        #         axs[0, 1].imshow(instance_pred, cmap='nipy_spectral')
        #         axs[0, 1].set_title(f"Instance Prediction")
        #         axs[0, 1].axis('off')
        #
        #         axs[1, 0].imshow(pred_edges_on_image)
        #         axs[1, 0].set_title(f"Overlay: GT (Green), Pred (Red)")
        #         axs[1, 0].axis('off')
        #
        #         axs[1, 1].imshow(gt_image, cmap='nipy_spectral')
        #         axs[1, 1].set_title("Ground Truth")
        #         axs[1, 1].axis('off')
        #         self.trainer.logger.experiment.log({"for presentation": wandb.Image(fig)})
        #         plt.close(fig)




        # plot_2d_images(self.preds_signDist_list[0], self.test_sign_dist_targets[0], self.targets_list[0])

        self.preds_signDist_list.clear()
        self.preds_bin_list.clear()
        self.targets_list.clear()
        self.test_sign_dist_targets.clear()
        self.data_list.clear()
        self.datasets_list.clear()
        self.mask_name_list.clear()

    # # for presentation
    # def test_step(self, batch, batch_idx):
    #     self.device1111 = batch[0].device
    #     return 0
    # def on_test_epoch_end(self):
    #     # sim_gt_sign_dist = self.signed_distance_function(
    #     #     torch.from_numpy(self.mask_test_sim).unsqueeze(0)).unsqueeze(0)
    #     sim_sign_dist_pred, sim_bin_seg_pred = self.model(
    #         torch.from_numpy(
    #             (self.image_test_sim - self.image_test_sim.mean()) / (self.image_test_sim.std())).unsqueeze(
    #             0).unsqueeze(0).to(self.device1111))
    #     if self.kernel_type == 'sigmoid':
    #         sim_edge_map = self.detect_edges_sigmoid(sim_bin_seg_pred)
    #     sim_edge_map_np = sim_edge_map[0, 0].cpu().numpy()
    #     # sim_pred_edges_on_image = self.overlay_edges_on_grayscale(self.image_test_sim,
    #     #                                                           sim_edge_map_np,
    #     #                                                           (sim_gt_sign_dist == 0)[0, 0].cpu().numpy())
    #     # sim_mhd_loss_gt = self.modified_hausdorff_distance(sim_bin_seg_pred,
    #     #                                                    (sim_gt_sign_dist == 0))
    #     # sim_mhd_loss_pred = self.modified_hausdorff_distance(sim_gt_sign_dist, sim_edge_map)
    #     seg_scores, det_scores = check_accuracy([sim_sign_dist_pred], [sim_bin_seg_pred],
    #                                     [torch.from_numpy(self.mask_test_sim).unsqueeze(0).unsqueeze(0).to(
    #                                         self.device1111)],
    #                                     batch_size=1, three_d=False, _2_5d=False)
    #     instance_pred, _ = get_cell_instances((torch.sigmoid(sim_bin_seg_pred) > 0.5)[0, 0].cpu().numpy(), three_d=False)
    #     instance_pred[instance_pred != 0] += 5
    #     # fig, axs = plt.subplots()
    #     # axs.imshow(sim_pred_edges_on_image)
    #     # axs.axis("off")
    #     # axs.set_title(f"mhd_pred: {sim_mhd_loss_pred:.2f} | mhd_gt: {sim_mhd_loss_gt:.2f} | SEG: {seg_scores:.2f}")
    #     # self.logger.experiment.log(
    #     #     {f"sim+ (red-prediction, green-gt)": wandb.Image(fig)})
    #     # plt.close(fig)
    #
    #
    #     fig, axs = plt.subplots(2, 5, figsize=(15, 5))
    #     # axs[0, 0].imshow(sim_pred_edges_on_image)
    #     # axs[0, 0].set_title("Input Image")
    #     # axs[0, 0].axis('off')
    #
    #     im0 = axs[0, 1].imshow(sim_sign_dist_pred[0, 0].cpu().numpy(), cmap='jet')
    #     axs[0, 1].set_title("SD Prediction")
    #     axs[0, 1].axis('off')
    #     plt.colorbar(im0, ax=axs[0, 1], orientation='vertical', fraction=0.046, pad=0.04)
    #
    #     im1 = axs[0, 2].imshow(torch.sigmoid(sim_bin_seg_pred)[0, 0].cpu().numpy(), cmap='jet')
    #     axs[0, 2].set_title("Soft Binary Prediction")
    #     axs[0, 2].axis('off')
    #     plt.colorbar(im1, ax=axs[0, 2], orientation='vertical', fraction=0.046, pad=0.04)
    #
    #     # axs[0, 3].imshow(bin_pred_hard_0_5, cmap='viridis')
    #     # axs[0, 3].set_title(f"Binary Prediction > 0.5")
    #     # axs[0, 3].axis('off')
    #
    #     axs[0, 4].imshow(instance_pred, cmap='nipy_spectral')
    #     axs[0, 4].set_title(f"Instance Prediction")
    #     axs[0, 4].axis('off')
    #
    #     im1 = axs[1, 0].imshow(1-torch.sigmoid(sim_bin_seg_pred)[0, 0].cpu().numpy(), cmap='jet')
    #     axs[1, 0].set_title("Inverse Soft Binary Prediction")
    #     axs[1, 0].axis('off')
    #     plt.colorbar(im1, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
    #
    #     im2 = axs[1, 1].imshow(sim_edge_map_np, cmap='jet')
    #     axs[1, 1].set_title("Prediction Edges Map")
    #     axs[1, 1].axis('off')
    #     plt.colorbar(im2, ax=axs[1, 1], orientation='vertical', fraction=0.046, pad=0.04)
    #
    #     # axs[1, 2].imshow(self.test_sign_dist_targets[0][0, 0].cpu().numpy(), cmap='jet')
    #     # axs[1, 2].set_title("SD Ground Truth")
    #     # axs[1, 2].axis('off')
    #
    #     # axs[1, 3].imshow(gt_image > 0, cmap='viridis')
    #     # axs[1, 3].set_title("Semantic Ground Truth")
    #     # axs[1, 3].axis('off')
    #     #
    #     # axs[1, 4].imshow(gt_image, cmap='nipy_spectral')
    #     # axs[1, 4].set_title("Ground Truth")
    #     # axs[1, 4].axis('off')
    #
    #     plt.tight_layout()
    #
    #     self.trainer.logger.experiment.log({"image": wandb.Image(fig)})
    #     plt.close(fig)


class Unet2dWrap_CE(BaseModel):
    def __init__(self, model_cfg, criterion, optimizer_params, scheduler_params, batch_size, pre_trained_path):
        super().__init__(criterion, optimizer_params, scheduler_params)

        self.batch_size = batch_size
        self.l1_lambda = model_cfg.pop('l1_lambda') if 'l1_lambda' in model_cfg else None
        if model_cfg.pop('dense'):
            self.model = DenseUnet2d(model_cfg=model_cfg, pre_trained_path=pre_trained_path)
        else:
            self.model = Unet2d(model_cfg=model_cfg, pre_trained_path=pre_trained_path)
            # self.model = UNET()
        self.preds = []
        self.targets_list = []
        self.data_list = []
        self.preds_train = []
        self.targets_list_train = []

    @staticmethod
    def detect_edges(mask, threshold=0.25):
        # Compute the gradients along rows and columns
        gradient_x = torch.gradient(mask, dim=0)[0]
        gradient_y = torch.gradient(mask, dim=1)[0]

        gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        masked_gradient_magnitude = gradient_magnitude * mask
        edge_mask = (masked_gradient_magnitude > threshold).to(torch.int)

        return edge_mask

    def split_mask(self, mask):
        three_classes_mask = torch.zeros_like(mask, dtype=torch.int32)
        for batch_idx in range(mask.size()[0]):
            unique_elements = torch.unique(mask[batch_idx].flatten())
            for element in unique_elements:
                if element != 0:
                    element_mask = (mask[batch_idx] == element).to(torch.int)
                    edges = self.detect_edges(element_mask)
                    element_mask -= edges
                    three_classes_mask[batch_idx][edges == 1] = 1
                    three_classes_mask[batch_idx][element_mask == 1] = 2

        return three_classes_mask.squeeze(1).to(mask.device).long()

    def calc_loss(self, pred_seg, classes_targets):
        def calcL1Loss():
            l1_loss = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
            return self.l1_lambda * l1_loss

        loss = 0
        for loss_fn in self.criterion:
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                if hasattr(loss_fn, 'weight') and loss_fn.weight is not None:
                    loss_fn.weight = loss_fn.weight.to(pred_seg.device)
                loss += loss_fn(pred_seg, classes_targets)
        if self.l1_lambda is not None:
            loss += calcL1Loss()  # add l1 loss to the total loss
        return loss

    def prepare_batch(self, batch):
        data, original_targets, _ = batch
        classes_targets = self.split_mask(original_targets)
        return data, classes_targets, original_targets

    def predict_batch(self, batch):
        data, classes_targets, original_targets = self.prepare_batch(batch)
        pred_seg, _ = self.model(data)
        return data, pred_seg, classes_targets, original_targets

    def training_step(self, batch, batch_idx):
        data, pred_seg, classes_targets, original_targets = self.predict_batch(batch)
        loss = self.calc_loss(pred_seg, classes_targets)
        self.log('train_loss', loss, prog_bar=True)
        if (self.current_epoch + 1) % 5 == 0:
            self.preds_train.append(pred_seg)
            self.targets_list_train.append(original_targets)

        return loss

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 5 == 0:
            print(f"\nCalculating train Jaccard Index for epoch: {self.current_epoch}")

            jaccard, _ = check_accuracy_CE(self.preds_train, self.targets_list_train, self.batch_size, three_d=False,
                                           _2_5d=False)
            self.log("train_SEG", jaccard, on_epoch=True, prog_bar=True)
            self.preds_train.clear()
            self.targets_list_train.clear()

    def validation_step(self, batch, batch_idx):
        data, pred_seg, classes_targets, original_targets = self.predict_batch(batch)
        loss = self.calc_loss(pred_seg, classes_targets)
        self.log("val_loss", loss, batch_size=batch[0].shape[0], on_epoch=True, on_step=False)
        if (self.current_epoch + 1) % 5 == 0:
            self.preds.append(pred_seg)
            self.targets_list.append(original_targets)
            self.data_list.append(data)
        return loss

    def on_validation_epoch_end(self):
        def predict_classes(preds):
            preds_softmax = F.softmax(preds, dim=1)  # Apply softmax along the class dimension
            _, predicted_classes = torch.max(preds_softmax, dim=1)  # Get the index of the maximum probability
            return predicted_classes

        if (self.current_epoch + 1) % 5 == 0:
            print(f"\nCalculating Jaccard Index for epoch: {self.current_epoch}")
            jaccard, _ = check_accuracy_CE(self.preds, self.targets_list, self.batch_size, three_d=False, _2_5d=False)
            self.log("val_SEG", jaccard, on_epoch=True, prog_bar=True)

            predicted_classes = predict_classes(self.preds[0])[0].cpu().numpy()

            foreground_mask = (predicted_classes == 2)
            pred_labels_mask, _ = get_cell_instances(foreground_mask, three_d=False)
            pred_labels_mask[pred_labels_mask != 0] += 5
            # pred_labels_mask = watershed_labels_from_binary(foreground_mask)

            # separator = np.ones((predicted_classes.shape[0], 10), dtype=np.uint8)
            # combined_image = np.hstack((predicted_classes * torch.max(self.targets_list[0][0]).item(), separator,
            #                             pred_labels_mask, separator, self.targets_list[0][0].cpu().numpy()))
            # self.trainer.logger.experiment.log({"image": wandb.Image(combined_image)})
            fig, axs = plt.subplots(2, 2, figsize=(15, 5))
            axs[0, 0].imshow(self.data_list[0][0, 0].cpu().numpy(), cmap='gray')
            axs[0, 0].set_title("Input Image")
            axs[0, 0].axis('off')

            axs[0, 1].imshow(pred_labels_mask, cmap='nipy_spectral')
            axs[0, 1].set_title("Instance Prediction")
            axs[0, 1].axis('off')

            axs[1, 0].imshow(predicted_classes, cmap='jet')
            axs[1, 0].set_title("Predicted Classes")
            axs[1, 0].axis('off')

            axs[1, 1].imshow(self.targets_list[0][0].cpu().numpy(), cmap='nipy_spectral')
            axs[1, 1].set_title("Ground Truth")
            axs[1, 1].axis('off')

            plt.tight_layout()

            self.trainer.logger.experiment.log({"image": wandb.Image(fig)})
            plt.close(fig)

            self.preds.clear()
            self.targets_list.clear()
            self.data_list.clear()

    def test_step(self, batch, batch_idx):
        data, pred_seg, classes_targets, original_targets = self.predict_batch(batch)
        loss = self.calc_loss(pred_seg, classes_targets)
        self.log("test_loss", loss, batch_size=batch[0].shape[0], on_epoch=True, on_step=False)

        self.preds.append(pred_seg)
        self.targets_list.append(original_targets)
        return loss

    def on_test_epoch_end(self):
        def predict_classes(preds):
            preds_softmax = F.softmax(preds, dim=1)  # Apply softmax along the class dimension
            _, predicted_classes = torch.max(preds_softmax, dim=1)  # Get the index of the maximum probability
            return predicted_classes

        print(f"Calculating Jaccard Index for epoch {self.current_epoch + 1}")

        jaccard, std = check_accuracy_CE(self.preds, self.targets_list, self.batch_size, three_d=False, _2_5d=False)
        # self.log("jaccard_index", jaccard, on_epoch=True, prog_bar=True)
        # self.log("jaccard_index_STD", std, on_epoch=True, prog_bar=True)
        #
        # # plot_2d_images(self.test_preds_signDist[0], self.test_preds_markers[0],
        # #                self.test_sign_dist_targets[0], self.test_targets[0])
        # predicted_classes = predict_classes(self.preds[5]).cpu().numpy()
        # import matplotlib.pyplot as plt
        # ig, axs = plt.subplots(1, 3, figsize=(10, 10))
        # axs[0].imshow(predicted_classes[0], cmap='gray')
        # axs[0].set_title('pred_signdist')
        # axs[0].axis('off')
        # axs[1].imshow((predicted_classes[0] == 2), cmap='gray')
        # axs[1].set_title('pred_signdist')
        # axs[1].axis('off')
        # axs[2].imshow(self.targets_list[5][0].cpu().numpy(), cmap='gray')
        # axs[2].set_title('gt_signdist')
        # axs[2].axis('off')
        # plt.tight_layout()
        # plt.show()

        self.preds.clear()
        self.targets_list.clear()

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)


# import torchvision.transforms.functional as TF
# class UNET(nn.Module):
#     def __init__(
#             self, in_channels=1, out_channels=3, features=[64, 128, 256, 512, 1024],
#     ):
#         super(UNET, self).__init__()
#         self.ups = nn.ModuleList()
#         self.downs = nn.ModuleList()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # Down part of UNET
#         for feature in features:
#             self.downs.append(DoubleConv(in_channels, feature))
#             in_channels = feature
#
#         # Up part of UNET
#         for feature in reversed(features):
#             self.ups.append(
#                 nn.ConvTranspose2d(
#                     feature * 2, feature, kernel_size=2, stride=2,
#                 )
#             )
#             self.ups.append(DoubleConv(feature * 2, feature))
#
#         self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
#         self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
#
#     def forward(self, x):
#         skip_connections = []
#
#         for down in self.downs:
#             x = down(x)
#             skip_connections.append(x)
#             x = self.pool(x)
#
#         x = self.bottleneck(x)
#         skip_connections = skip_connections[::-1]
#
#         for idx in range(0, len(self.ups), 2):
#             x = self.ups[idx](x)
#             skip_connection = skip_connections[idx // 2]
#
#             if x.shape != skip_connection.shape:
#                 x = TF.resize(x, size=skip_connection.shape[2:])
#
#             concat_skip = torch.cat((skip_connection, x), dim=1)
#             x = self.ups[idx + 1](concat_skip)
#
#         return self.final_conv(x), None


class Unet2_5dWrap(BaseModel):
    def __init__(self, model_cfg, criterion, optimizer_params, scheduler_params, batch_size, pre_trained_path):
        super().__init__(criterion, optimizer_params, scheduler_params)

        self.batch_size = batch_size
        self.alpha_mhd_gt = model_cfg.pop('alpha_mhd')
        self.alpha_mhd_pred = 0
        self.alpha_mse = model_cfg.pop('alpha_mse')
        self.use_input_adapter = model_cfg.pop('use_input_adapter', False)
        self.modify_first_conv = model_cfg.pop('modify_first_conv', False)
        if model_cfg.with_markers and 'sign_dist_leg_warmup' in model_cfg:
            self.sign_dist_leg_warmup = model_cfg.pop('sign_dist_leg_warmup')
        else:
            self.sign_dist_leg_warmup = 0

        self.l1_lambda = model_cfg.pop('l1_lambda') if 'l1_lambda' in model_cfg else None
        # if model_cfg.pop('dense'):
        #     self.model = DenseUnet2d(model_cfg=model_cfg, pre_trained_path=pre_trained_path)

        if self.use_input_adapter:
            self.model = Unet2_5d_Adapter(model_cfg=model_cfg, pre_trained_path=pre_trained_path)
        elif self.modify_first_conv:
            self.model = Unet2_5d_Modify(model_cfg=model_cfg, pre_trained_path=pre_trained_path)
        else:
            self.model = Unet2d_One_Leg(model_cfg=model_cfg, pre_trained_path=pre_trained_path)

        self.preds_signDist_list = []
        self.preds_bin_list = []
        self.targets_list = []
        self.test_sign_dist_targets = []
        self.data_list = []
        self.datasets_list = []
        self.preds_signDist_list_train = []
        self.preds_bin_list_train = []
        self.targets_list_train = []
        self.datasets_list_train = []

    def signed_distance_function(self, tensor):
        """
        Computes the signed distance function (SDF) for a given tensor.

        Args:
            tensor (torch.Tensor): A binary or multi-class tensor.

        Returns:
            torch.Tensor: A tensor of the same shape with the signed distances.
        """
        np_array = tensor.cpu().numpy().astype(np.bool_)

        # outside_distances = np.maximum(distance_transform_edt(np_array == 0) - 1, 0)  # remove 1 to set the edges as 0
        # inside_distances = np.maximum(distance_transform_edt(np_array > 0) - 1, 0)  # remove 1 to set the edges as 0
        #
        # # SD = inside distance (positive) - outside distance (negative)
        # sd = inside_distances - outside_distances
        sd = np.zeros_like(np_array, dtype=np.float32)
        for i in np.unique(np_array):
            if i == 0:
                sd -= np.maximum(distance_transform_edt(np_array == i) - 1,
                                 0)  # remove 1 to set the edges as 0
            else:
                sd += np.maximum(distance_transform_edt(np_array == i) - 1,
                                 0)  # remove 1 to set the edges as 0

        sd_tensor = torch.from_numpy(sd).float().to(tensor.device)
        if self.use_input_adapter or self.modify_first_conv:
            sd_tensor = (2 * torch.sigmoid(self.model.model.alpha * sd_tensor)) - 1
        else:
            sd_tensor = (2 * torch.sigmoid(self.model.alpha * sd_tensor)) - 1
        return sd_tensor

    def batch_signed_distance_function(self, tensor):
        """
        Applies the signed distance function to each 2D slice in the 5D tensor.

        Args:
            tensor (torch.Tensor): A 5D tensor of shape [B, 1, D, H, W].

        Returns:
            torch.Tensor: A 4D tensor with the same shape, containing the SD for each 2D slice.
        """

        B, _, D, H, W = tensor.shape
        sd_tensor = torch.zeros_like(tensor).to(tensor.device)

        for b in range(B):
            for d in range(D):
                sd_slice = self.signed_distance_function(tensor[b, 0, d])
                sd_tensor[b, 0, d] = sd_slice

        return sd_tensor

    def remove_touching_boundary(self, targets):
        """
        Removes 1-pixel-wide touching boundaries between labeled regions
        for a batch of 2D labeled images of shape [B, 1, D, H, W].
        """
        device = targets.device
        B, _, D, H, W = targets.shape

        kernel = torch.tensor([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        results = torch.zeros_like(targets)

        for b in range(B):
            for d in range(D):
                target = targets[b, 0, d]
                unique_labels = torch.unique(target)
                edge_mask = torch.zeros_like(target, dtype=torch.bool)

                for i, current_label in enumerate(unique_labels):
                    if current_label == 0:
                        continue
                    binary = (target == current_label).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    dilated = F.conv2d(F.pad(binary, (1, 1, 1, 1), mode='constant', value=0), kernel)
                    neighbors = dilated[0, 0]  # [H, W]

                    # Where dilation overlaps with *other* labels
                    touching = (neighbors > 0) & (target != current_label) & (target != 0)
                    edge_mask |= touching

                cleaned = target.clone()
                cleaned[edge_mask] = 0
                results[b, 0, d] = cleaned

        return results

    def normalize_batch_2_5d(self, data):
        """
        Normalize each [3, H, W] triplet in the batch to zero mean and unit std.

        Args:
            data: Tensor of shape (N, 3, H, W)

        Returns:
            Tensor of same shape, normalized
        """
        mean = data.mean(dim=(1, 2, 3), keepdim=True)  # shape: [N, 1, 1, 1]
        std = data.std(dim=(1, 2, 3), keepdim=True)  # shape: [N, 1, 1, 1]
        normalized = (data - mean) / (std + 1e-8)
        return normalized

    def calcL1Loss(self):
        l1_loss = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
        return self.l1_lambda * l1_loss

    def modified_hausdorff_distance(self, sign_dist, edge_mask):
        return torch.sum(torch.abs(sign_dist * edge_mask)) / float(torch.sum(edge_mask).item())


    def detect_edges_sigmoid(self, bin_seg_pred):
        sig_bin_seg_pred = torch.sigmoid(bin_seg_pred)
        edge_mask = sig_bin_seg_pred * (1 - sig_bin_seg_pred)
        batch_min = edge_mask.amin(dim=(-2, -1), keepdim=True)
        batch_max = edge_mask.amax(dim=(-2, -1), keepdim=True)

        normalized_edge_mask = (edge_mask - batch_min) / (batch_max - batch_min + 1e-8)
        return normalized_edge_mask


    def calc_loss(self, sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets):
        l1_loss = 0
        mse_loss = 0
        bce_loss = 0
        mse_bin_loss = 0
        mhd_loss_pred = 0
        mhd_loss_gt = 0
        sig_bin_seg_pred = torch.sigmoid(bin_seg_pred)
        for loss_fn in self.criterion:
            if isinstance(loss_fn, nn.MSELoss) or isinstance(loss_fn, nn.SmoothL1Loss):
                mse_loss = self.alpha_mse * loss_fn(sign_dist_pred,
                                                    sign_dist_targets)  # torch.sigmoid(bin_seg_pred)

                # mse_bin_loss = 5 * loss_fn(torch.sigmoid(bin_seg_pred), (sign_dist_targets + 1) / 2)
                mse_bin_loss = 5 * loss_fn(torch.sigmoid(bin_seg_pred), bin_targets)
            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                # bce_loss = loss_fn(bin_seg_pred, (sign_dist_targets + 1) / 2)
                bce_loss = loss_fn(bin_seg_pred, bin_targets)

        mhd_loss_gt = self.alpha_mhd_gt * self.modified_hausdorff_distance(sign_dist_pred, (
                sign_dist_targets == 0))  # sig_bin_seg_pred

        mhd_loss_pred = self.alpha_mhd_pred * self.modified_hausdorff_distance(sign_dist_targets,
                                                                               self.detect_edges_sigmoid(
                                                                                   bin_seg_pred)
                                                                               )

        if self.l1_lambda is not None:
            l1_loss = self.calcL1Loss()  # start after getting the loss from sign dist leg

        return (mse_loss + bce_loss + mhd_loss_gt + mhd_loss_pred + l1_loss + mse_bin_loss, mse_loss, bce_loss,
                mhd_loss_gt, mhd_loss_pred, l1_loss, mse_bin_loss)


    def overlay_edges_on_grayscale(self, grayscale_img, pred_edge_map, gt_edge_map):
        """
        Overlays predicted and ground truth edge maps on the grayscale image.

        Args:
            grayscale_img (numpy array): Grayscale image of shape [H, W].
            pred_edge_map (numpy array): Predicted edge map of shape [H, W].
            gt_edge_map (numpy array): Ground truth edge map of shape [H, W].

        Returns:
            RGB image with edges overlayed
        """
        grayscale_img = ((grayscale_img - grayscale_img.min()) / (
                grayscale_img.max() - grayscale_img.min()) * 255).astype(np.uint8)

        rgb_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
        max_val = np.iinfo(grayscale_img.dtype).max
        rgb_img[pred_edge_map >= 0.85, 0] = max_val  # Set red channel
        rgb_img[gt_edge_map == 1, 1] = max_val  # Set green channel

        return rgb_img

    def get_scores_by_dataset(self, datasets, seg_scores):
        scores_by_dataset = EasyDict()
        for ds, score in zip(datasets, seg_scores):
            if ds not in scores_by_dataset:
                scores_by_dataset[ds] = []
            scores_by_dataset[ds].append(score)
        return scores_by_dataset

    def prepare_batch(self, batch):
        data, original_targets, _ = batch
        # original_targets = original_targets

        # chose
        # targets = original_targets
        targets = self.remove_touching_boundary(original_targets)

        sign_dist_targets = self.batch_signed_distance_function(targets)

        bin_targets = (targets > 0).float()

        data = convert_3d_to_2_5d(data)
        data = self.normalize_batch_2_5d(data)
        targets_shape = targets.shape
        sign_dist_targets = sign_dist_targets.squeeze(1).view(targets_shape[0]*targets_shape[2], 1, *targets_shape[3:])
        bin_targets = bin_targets.squeeze(1).view(targets_shape[0]*targets_shape[2], 1, *targets_shape[3:])
        return data, sign_dist_targets, bin_targets, original_targets

    def predict_batch(self, batch):
        data, sign_dist_targets, bin_targets, original_targets = self.prepare_batch(batch)
        sign_dist_pred, bin_seg_pred = self.model(data)
        return sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets, original_targets

    def on_train_epoch_start(self):
        if self.current_epoch == self.sign_dist_leg_warmup:
            self.alpha_mhd_pred = self.alpha_mhd_gt
            # reset scheduler
            if hasattr(self, 'reduce_lr_scheduler'):
                if isinstance(self.reduce_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.reduce_lr_scheduler.best = self.reduce_lr_scheduler.mode_worse
                    self.reduce_lr_scheduler.cooldown_counter = 0
                    self.reduce_lr_scheduler.num_bad_epochs = 0
        elif self.current_epoch < self.sign_dist_leg_warmup and self.current_epoch % 10 == 0: #todo uncoment when not on harvard
            if hasattr(self, 'reduce_lr_scheduler'):
                if isinstance(self.reduce_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.reduce_lr_scheduler.best = self.reduce_lr_scheduler.mode_worse
                    self.reduce_lr_scheduler.cooldown_counter = 0
                    self.reduce_lr_scheduler.num_bad_epochs = 0


    def training_step(self, batch, batch_idx):
        sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets, original_targets = self.predict_batch(batch)
        if (self.current_epoch + 1) % 5 == 0:
            self.preds_signDist_list_train.append(sign_dist_pred)
            self.preds_bin_list_train.append(bin_seg_pred)
            self.targets_list_train.append(original_targets)
            self.datasets_list_train += batch[-1]
        loss, mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, l1_loss, mse_bin_loss = self.calc_loss(
            sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

        self.log('train_mse_loss', mse_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_mse_bin_loss', mse_bin_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_bce_loss', bce_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_mhd_loss SD_pred*(gt==0)', mhd_loss_gt, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_mhd_loss (pred==0)*SD_GT', mhd_loss_pred, prog_bar=False, on_epoch=True, on_step=False)
        self.log('train_l1_loss', l1_loss, prog_bar=False, on_epoch=True, on_step=False)
        if batch_idx == 1:
            data = batch[0]
            data = convert_3d_to_2_5d(data)
            data = self.normalize_batch_2_5d(data)
            fig, axs = plt.subplots(3)
            for i in range(3):
                im = axs[i].imshow(data[i + 7, 1].cpu().numpy(), cmap="gray")
                axs[i].set_title(f"mean: {data[i + 7].mean():.3f}, std: {data[i + 7].std():.3f}")
                axs[i].axis('off')
                plt.colorbar(im, ax=axs[i], orientation='vertical', fraction=0.046, pad=0.04)
            plt.savefig(f"222222222222222")
        return loss


    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 5 == 0:
            print(f"\nCalculating train Jaccard Index for epoch: {self.current_epoch}")

            seg_scores, det_score = check_accuracy(self.preds_signDist_list_train, self.preds_bin_list_train,
                                        self.targets_list_train,
                                        batch_size=self.batch_size, three_d=False, _2_5d=True)

            self.log("train_SEG", seg_scores.mean(), on_epoch=True, prog_bar=True)
            print(f"total train SEG: mean={seg_scores.mean():.3f}, std={seg_scores.std():.3f}")
            scores_by_dataset = self.get_scores_by_dataset(self.datasets_list_train, seg_scores)
            for ds, scores in scores_by_dataset.items():
                scores = np.array(scores)
                mean = scores.mean()
                std = scores.std()
                print(f"{ds}: mean={mean:.3f}, std={std:.3f}")
                self.log(f"train_SEG_{ds}", mean, on_epoch=True, prog_bar=True)
            self.preds_signDist_list_train.clear()
            self.preds_bin_list_train.clear()
            self.targets_list_train.clear()
            self.datasets_list_train.clear()

    def validation_step(self, batch, batch_idx):
        sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets, original_targets = self.predict_batch(batch)
        if (self.current_epoch + 1) % 5 == 0:
            self.preds_signDist_list.append(sign_dist_pred)
            self.preds_bin_list.append(bin_seg_pred)
            self.targets_list.append(original_targets)
            self.datasets_list += batch[-1]
            if not self.test_sign_dist_targets:
                self.test_sign_dist_targets.append(sign_dist_targets)
                self.data_list.append(batch[0])

        loss, mse_loss, bce_loss, mhd_loss_gt, mhd_loss_pred, l1_loss, mse_bin_loss = self.calc_loss(
            sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets)
        self.log("val_loss", loss, batch_size=batch[0].shape[0], on_epoch=True, on_step=False)

        self.log('val_mse_loss', mse_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_mse_bin_loss', mse_bin_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_bce_loss', bce_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_mhd_loss SD_pred*(gt==0)', mhd_loss_gt, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_mhd_loss (pred==0)*SD_GT', mhd_loss_pred, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val_l1_loss', l1_loss, prog_bar=False, on_epoch=True, on_step=False)
        if batch_idx == 1:
            data = batch[0]
            data = convert_3d_to_2_5d(data)
            data = self.normalize_batch_2_5d(data)
            fig, axs = plt.subplots(3)
            for i in range(3):
                im = axs[i].imshow(data[i+7, 1].cpu().numpy(), cmap="gray")
                axs[i].set_title(f"mean: {data[i + 7].mean():.3f}, std: {data[i + 7].std():.3f}")
                axs[i].axis('off')
                plt.colorbar(im, ax=axs[i], orientation='vertical', fraction=0.046, pad=0.04)
            plt.savefig("333333333333333")
        return loss



    def on_validation_epoch_end(self):
        if self.use_input_adapter or self.modify_first_conv:
            self.log("alpha", self.model.model.alpha.item(), prog_bar=False)
            self.log("beta", self.model.model.beta.item(), prog_bar=False)
        else:
            self.log("alpha", self.model.alpha.item(), prog_bar=False)
            self.log("beta", self.model.beta.item(), prog_bar=False)

        if (self.current_epoch + 1) % 5 == 0:
            print(f"\nCalculating val Jaccard Index for epoch: {self.current_epoch}")

            seg_scores, det_scores = check_accuracy(self.preds_signDist_list, self.preds_bin_list, self.targets_list,
                                        batch_size=self.batch_size, three_d=False, _2_5d=True)
            self.log("val_SEG", seg_scores.mean(), on_epoch=True, prog_bar=True)
            print(f"total val SEG: mean={seg_scores.mean():.3f}, std={seg_scores.std():.3f}")
            scores_by_dataset = self.get_scores_by_dataset(self.datasets_list, seg_scores)
            for ds, scores in scores_by_dataset.items():
                scores = np.array(scores)
                mean = scores.mean()
                std = scores.std()
                print(f"{ds}: mean={mean:.3f}, std={std:.3f}")
                self.log(f"val_SEG_{ds}", mean, on_epoch=True, prog_bar=True)
            middle_slice = self.data_list[0].shape[0] // 2 + 2
            input_image = self.data_list[0][0, 0, middle_slice].cpu().numpy()
            sd_image = self.preds_signDist_list[0][middle_slice, 0].cpu().numpy()
            gt_image = self.targets_list[0][0, 0, middle_slice].cpu().numpy()

            edge_map = self.detect_edges_sigmoid(self.preds_bin_list[0])
            bin_pred_hard_0_5 = (torch.sigmoid(self.preds_bin_list[0]) > 0.5)[middle_slice, 0].cpu().numpy()
            bin_pred_soft = torch.sigmoid(self.preds_bin_list[0])[middle_slice, 0].cpu().numpy()
            inv_bin_pred_soft = 1 - bin_pred_soft
            edge_map_np = edge_map[middle_slice, 0].cpu().numpy()

            instance_pred, _ = get_cell_instances(bin_pred_hard_0_5, three_d=False)
            instance_pred[instance_pred != 0] += 5

            fig, axs = plt.subplots(2, 5, figsize=(15, 5))
            axs[0, 0].imshow(input_image, cmap='gray')
            axs[0, 0].set_title("Input Image")
            axs[0, 0].axis('off')

            im0 = axs[0, 1].imshow(sd_image, cmap='jet')
            axs[0, 1].set_title("SD Prediction")
            axs[0, 1].axis('off')
            plt.colorbar(im0, ax=axs[0, 1], orientation='vertical', fraction=0.046, pad=0.04)

            im1 = axs[0, 2].imshow(bin_pred_soft, cmap='jet')
            axs[0, 2].set_title("Soft Binary Prediction")
            axs[0, 2].axis('off')
            plt.colorbar(im1, ax=axs[0, 2], orientation='vertical', fraction=0.046, pad=0.04)

            axs[0, 3].imshow(bin_pred_hard_0_5, cmap='viridis')
            axs[0, 3].set_title(f"Binary Prediction > 0.5")
            axs[0, 3].axis('off')

            axs[0, 4].imshow(instance_pred, cmap='nipy_spectral')
            axs[0, 4].set_title(f"Instance Prediction")
            axs[0, 4].axis('off')

            im1 = axs[1, 0].imshow(inv_bin_pred_soft, cmap='jet')
            axs[1, 0].set_title("Inverse Soft Binary Prediction")
            axs[1, 0].axis('off')
            plt.colorbar(im1, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)

            im2 = axs[1, 1].imshow(edge_map_np, cmap='jet')
            axs[1, 1].set_title("Prediction Edges Map")
            axs[1, 1].axis('off')
            plt.colorbar(im2, ax=axs[1, 1], orientation='vertical', fraction=0.046, pad=0.04)

            im3 = axs[1, 2].imshow(self.test_sign_dist_targets[0][middle_slice, 0].cpu().numpy(), cmap='jet')
            axs[1, 2].set_title("SD Ground Truth")
            axs[1, 2].axis('off')
            plt.colorbar(im3, ax=axs[1, 2], orientation='vertical', fraction=0.046, pad=0.04)

            axs[1, 3].imshow(gt_image > 0, cmap='viridis')
            axs[1, 3].set_title("Semantic Ground Truth")
            axs[1, 3].axis('off')

            axs[1, 4].imshow(gt_image, cmap='nipy_spectral')
            axs[1, 4].set_title("Ground Truth")
            axs[1, 4].axis('off')

            plt.tight_layout()

            self.trainer.logger.experiment.log({"image": wandb.Image(fig)})
            plt.close(fig)

            self.preds_signDist_list.clear()
            self.preds_bin_list.clear()
            self.targets_list.clear()
            self.test_sign_dist_targets.clear()
            self.data_list.clear()
            self.datasets_list.clear()

    def test_step(self, batch, batch_idx):
        sign_dist_pred, bin_seg_pred, sign_dist_targets, bin_targets, original_targets = self.predict_batch(batch)

        self.preds_signDist_list.append(sign_dist_pred)
        self.preds_bin_list.append(bin_seg_pred)
        self.targets_list.append(original_targets)
        self.test_sign_dist_targets.append(sign_dist_targets)
        self.data_list.append(batch[0])
        self.datasets_list += batch[-1]

    def on_test_epoch_end(self):
        print("******************************************************")#todo
        print("todo")
        print("******************************************************")
        def overlay_edges_on_sdf(sd, edge_mask):
            overlay_tensor = sd.clone()  # Clone SDF tensor to avoid modifying the original
            overlay_tensor[edge_mask == 1] = sd.max() + 2
            return overlay_tensor[0, 0].cpu().numpy()

        print(f"\nCalculating Jaccard Index")

        seg_scores, det_scores = check_accuracy(self.preds_signDist_list, self.preds_bin_list, self.targets_list,
                                    batch_size=self.batch_size, three_d=False, _2_5d=True)

        self.log("test_SEG", seg_scores.mean(), on_epoch=True, prog_bar=True)
        print(f"total test SEG: mean={seg_scores.mean():.3f}, std={seg_scores.std():.3f}")
        print(f"total test DEG: mean={det_scores.mean():.3f}, std={det_scores.std():.3f}")
        seg_scores_by_dataset = self.get_scores_by_dataset(self.datasets_list, seg_scores)
        det_scores_by_dataset = self.get_scores_by_dataset(self.datasets_list, det_scores)
        print("SEG:")
        for ds, scores in seg_scores_by_dataset.items():
            scores = np.array(scores)
            mean = scores.mean()
            std = scores.std()
            print(f"\t{ds}: mean={mean:.3f}, std={std:.3f}, len: {len(scores)}")
            self.log(f"test_SEG_{ds}", mean, on_epoch=True, prog_bar=True)
        print("DET:")
        for ds, scores in det_scores_by_dataset.items():
            scores = np.array(scores)
            mean = scores.mean()
            std = scores.std()
            print(f"\t{ds}: mean={mean:.3f}, std={std:.3f}, len: {len(scores)}")
            self.log(f"test_DEG_{ds}", mean, on_epoch=True, prog_bar=True)

        total_mse_loss = 0
        total_mhd_loss_gt = 0
        total_mhd_loss_pred = 0
        for i in range(len(self.preds_signDist_list)):
            mhd_loss_gt = self.modified_hausdorff_distance(self.preds_signDist_list[i],
                                                           (self.test_sign_dist_targets[i] == 0))

            # edge_signDist_pred = self.detect_edges_grad(self.preds_signDist_list[i])
            edge_signDist_pred = self.detect_edges_sigmoid(self.preds_bin_list[i])
            mhd_loss_pred = self.modified_hausdorff_distance(self.test_sign_dist_targets[i], edge_signDist_pred)

            if i < 10 or (i >= (len(self.preds_signDist_list) - 19) and i <= (len(self.preds_signDist_list) - 10)):
                input_image = self.data_list[i][0, 0].cpu().numpy()
                sd_image = self.preds_signDist_list[i][0, 0].cpu().numpy()
                gt_image = self.targets_list[i][0, 0].cpu().numpy()
                bin_pred = torch.sigmoid(self.preds_bin_list[i])[0, 0].cpu().numpy()
                semantic_pred = (torch.sigmoid(self.preds_bin_list[i]) > 0.5)[0, 0].cpu().numpy()
                instance_pred, _ = get_cell_instances(semantic_pred, three_d=False)
                instance_pred[instance_pred != 0] += 5

                # self.log('val_mhd_loss SD_pred*(gt==0)', float(f"{mhd_loss_gt:.3f}"))
                # self.log('val_mhd_loss (pred==0)*SD_GT', float(f"{mhd_loss_pred:.3f}"))

                wandb.log({'test_mhd_loss (pred==0)*SD_GT': float(f"{mhd_loss_pred:.3f}")})
                wandb.log({'test_mhd_loss SD_pred*(gt==0)': float(f"{mhd_loss_gt:.3f}")})

                fig, axs = plt.subplots(3, 5, figsize=(15, 5))
                axs[0, 0].imshow(input_image, cmap='gray')
                axs[0, 0].set_title("Input Image")
                axs[0, 0].axis('off')

                im0 = axs[0, 1].imshow(sd_image, cmap='jet')
                axs[0, 1].set_title("SD Prediction")
                axs[0, 1].axis('off')
                plt.colorbar(im0, ax=axs[0, 1], orientation='vertical', fraction=0.046, pad=0.04)

                im1 = axs[0, 2].imshow(bin_pred, cmap='jet')
                axs[0, 2].set_title("Soft Binary Prediction")
                axs[0, 2].axis('off')
                plt.colorbar(im1, ax=axs[0, 2], orientation='vertical', fraction=0.046, pad=0.04)

                axs[0, 3].imshow(semantic_pred, cmap='viridis')
                axs[0, 3].set_title(f"Binary Prediction > 0.5")
                axs[0, 3].axis('off')

                axs[0, 4].imshow(instance_pred, cmap='nipy_spectral')
                axs[0, 4].set_title(f"Instance Prediction")
                axs[0, 4].axis('off')

                im1 = axs[1, 0].imshow(1 - bin_pred, cmap='jet')
                axs[1, 0].set_title("Inverse Soft Binary Prediction")
                axs[1, 0].axis('off')
                plt.colorbar(im1, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)

                im2 = axs[1, 1].imshow(edge_signDist_pred[0, 0].cpu().numpy(), cmap='jet')
                axs[1, 1].set_title("Prediction Edges Map")
                axs[1, 1].axis('off')
                plt.colorbar(im2, ax=axs[1, 1], orientation='vertical', fraction=0.046, pad=0.04)

                axs[1, 2].imshow(self.test_sign_dist_targets[i][0, 0].cpu().numpy(), cmap='jet')
                axs[1, 2].set_title("SD Ground Truth")
                axs[1, 2].axis('off')

                axs[1, 3].imshow(gt_image > 0, cmap='viridis')
                axs[1, 3].set_title("Semantic Ground Truth")
                axs[1, 3].axis('off')

                axs[1, 4].imshow(gt_image, cmap='nipy_spectral')
                axs[1, 4].set_title("Ground Truth")
                axs[1, 4].axis('off')

                axs[2, 0].imshow((self.test_sign_dist_targets[i][0, 0] == 0).cpu().numpy(), cmap='jet')
                axs[2, 0].set_title("Ground Truth Edges Map")
                axs[2, 0].axis('off')
                plt.tight_layout()

                self.trainer.logger.experiment.log({"image": wandb.Image(fig)})
                plt.close(fig)

                pred_sd_gt_edges = overlay_edges_on_sdf(self.preds_signDist_list[i],
                                                        self.test_sign_dist_targets[i] == 0)
                pred_edges_gt_sd = overlay_edges_on_sdf(self.test_sign_dist_targets[i], edge_signDist_pred >= 0.85)

                fig, axs = plt.subplots(1, 2, figsize=(15, 5))
                im0 = axs[0].imshow(pred_sd_gt_edges, cmap='jet')
                axs[0].set_title(f"mhd_loss SD_pred*(gt==0) = {mhd_loss_gt:.3f}")
                axs[0].axis('off')
                plt.colorbar(im0, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)

                im1 = axs[1].imshow(pred_edges_gt_sd, cmap='jet')
                axs[1].set_title(f"mhd_loss (pred==0)*SD_GT = {mhd_loss_pred:.3f}")
                axs[1].axis('off')
                plt.colorbar(im1, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)

                plt.tight_layout()

                self.trainer.logger.experiment.log({"image_mhd": wandb.Image(fig)})
                plt.close(fig)

                pred_edges_on_image = self.overlay_edges_on_grayscale(input_image,
                                                                      edge_signDist_pred[0, 0].cpu().numpy(),
                                                                      (self.test_sign_dist_targets[i] == 0)[
                                                                          0, 0].cpu().numpy())

                seg_scores, det_scores = check_accuracy(self.preds_signDist_list[i], self.preds_bin_list[i],
                                            self.targets_list[i], batch_size=1,
                                            three_d=False, _2_5d=False)
                fig, axs = plt.subplots()
                axs.imshow(pred_edges_on_image)
                axs.axis("off")
                axs.set_title(
                    f"mhd_pred: {mhd_loss_pred:.2f} | mhd_gt: {mhd_loss_gt:.2f} | SEG: {seg_scores.mean():.2f}")
                self.logger.experiment.log(
                    {f"image with edge of prediction(red-prediction, green-gt)": wandb.Image(fig)})
                plt.close(fig)

                fig, axs = plt.subplots(2, 2)
                axs[0, 0].imshow(input_image, cmap='gray')
                axs[0, 0].set_title("Input Image")
                axs[0, 0].axis('off')

                axs[0, 1].imshow(instance_pred, cmap='nipy_spectral')
                axs[0, 1].set_title(f"Instance Prediction")
                axs[0, 1].axis('off')

                axs[1, 0].imshow(pred_edges_on_image)
                axs[1, 0].set_title(f"Overlay: GT (Green), Pred (Red)")
                axs[1, 0].axis('off')

                axs[1, 1].imshow(gt_image, cmap='nipy_spectral')
                axs[1, 1].set_title("Ground Truth")
                axs[1, 1].axis('off')
                self.trainer.logger.experiment.log({"for presentation": wandb.Image(fig)})
                plt.close(fig)

        self.preds_signDist_list.clear()
        self.preds_bin_list.clear()
        self.targets_list.clear()
        self.test_sign_dist_targets.clear()
        self.data_list.clear()
        self.datasets_list.clear()



