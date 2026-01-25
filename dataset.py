import lightning as pl
import os
import torch
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torchio as tio
import tifffile as tiff
from PIL import Image
from torchio.transforms import Transform
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from monai.transforms import NormalizeIntensity



class CropWithVoxelThreshold:
    def __init__(self, crop_size, threshold=5000, max_attempts=2000):
        self.crop_size = crop_size
        self.threshold = threshold
        self.max_attempts = max_attempts
        self.cropper = tio.CropOrPad(crop_size) if crop_size is not None else None

    def __call__(self, subj):
        if self.cropper is None:
            return subj

        attempts = 0
        while attempts < self.max_attempts:
            cropped_subj = self.cropper(subj)

            seg_tensor = cropped_subj.seg_mask.tensor
            num_voxels_above_zero = (seg_tensor > 0).sum().item()
            if num_voxels_above_zero > self.threshold:
                return cropped_subj

            attempts += 1

        raise ValueError(f"No valid crop found after {self.max_attempts} attempts.")

class SliceWiseNorm(Transform):
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        image = subject.image.tensor  # shape: (1, D, H, W)
        for i in range(image.shape[1]):
            slice_i = image[0, i]
            mean = slice_i.mean()
            std = slice_i.std()
            if std > 0:
                image[0, i] = (slice_i - mean) / std
        return subject

def parse_cfg_transform(transform_params):
    augs = transform_params.get('augmentations', None)
    if augs is None:
        return None

    aug_list = []
    for aug, params in augs.items():
        aug_func = getattr(tio, aug)
        aug_list.append(aug_func(p=0.5, **params))
    augmentations = tio.Compose(aug_list)
    return augmentations

class CELL3DDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.batch_size = data_cfg.batch_size
        self.num_workers = data_cfg.pop('num_workers')
        self.to_normalize = data_cfg.pop('to_normalize') if hasattr(data_cfg, 'to_normalize') else True
        self.harvard_csv = data_cfg.pop('harvard_csv') if hasattr(data_cfg, 'harvard_csv') else None
        img_size = data_cfg.img_size if hasattr(data_cfg, 'img_size') else None

        if data_cfg.sys == 'runai':
            self.base_dir = data_cfg.pop('dir_runai')
        elif data_cfg.sys == 'dgx':
            self.base_dir = data_cfg.pop('dir_dgx')
        else:
            raise ValueError("System type must be either 'runai' or 'dgx'.")
        if self.harvard_csv is None:
            self.train_dirs = [
                [os.path.join(self.base_dir, img) for img in data_cfg.train.img],
                [os.path.join(self.base_dir, seg) for seg in data_cfg.train.seg],
                # [os.path.join(self.base_dir, tra) for tra in data_cfg.train.tra]
            ]
            self.val_dirs = [
                [os.path.join(self.base_dir, img) for img in data_cfg.val.img],
                [os.path.join(self.base_dir, seg) for seg in data_cfg.val.seg],
                # [os.path.join(self.base_dir, tra) for tra in data_cfg.val.tra]
            ]
            self.test_dirs = [
                [os.path.join(self.base_dir, img) for img in data_cfg.test.img],
                [os.path.join(self.base_dir, seg) for seg in data_cfg.test.seg],
            ]
        else:
            self.train_dirs = None
            self.val_dirs = None
            self.test_dirs = None
        self.crop_transform = CropWithVoxelThreshold(img_size)
        self.transform = parse_cfg_transform(data_cfg.augmentations)

    def get_transform(self, train_aug):
        self.monai_norm = NormalizeIntensity(nonzero=False, channel_wise=False)
        self.tio_normalize = tio.Lambda(self.monai_norm, types_to_apply=[tio.INTENSITY])
        # self.tio_normalize = SliceWiseNorm()
        if self.to_normalize:
            if train_aug and self.transform is not None:
                transform = tio.Compose([self.tio_normalize, self.transform, self.crop_transform])
            else:
                transform = tio.Compose([self.tio_normalize, self.crop_transform])
        else:
            if train_aug and self.transform is not None:
                transform = tio.Compose([self.transform, self.crop_transform])
            else:
                transform = tio.Compose([self.crop_transform])
        return transform

    def setup(self, stage):

        if stage == 'fit':
            transform = self.get_transform(train_aug=True)
            self.train_ds = CELL3DDataset(
                image_paths=self.train_dirs,
                transform=transform,
                harvard_csv=self.harvard_csv,
                stage='train',
            )
        transform = self.get_transform(train_aug=False)
        self.val_ds = CELL3DDataset(
            image_paths=self.val_dirs,
            transform=transform,
            harvard_csv=self.harvard_csv,
            stage='val',
        )
        if stage == 'test':
            self.test_ds = CELL3DDataset(
                image_paths=self.test_dirs,
                transform=transform,
                harvard_csv=self.harvard_csv,
                stage=stage,
            )

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds, shuffle=True, batch_size=self.batch_size,
                              num_workers=self.num_workers, persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds, shuffle=False, batch_size=self.batch_size,
                   num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds, shuffle=False, batch_size=1,
                          num_workers=self.num_workers, persistent_workers=True)

    def predict_dataloader(self):
        raise NotImplementedError

class CELL3DDataset(Dataset):
    def __init__(self, image_paths, transform, harvard_csv, stage):
        self.image_paths = image_paths
        self.images = []
        self.seg = []
        self.datasets = []
        if harvard_csv:
            df = pd.read_csv(harvard_csv)
            for _, row in df.iterrows():

                if stage == 'train' and row["set"] == 'train':
                    self.images.append(row["image_path"])
                    self.seg.append(row["mask_path"])
                    self.datasets.append("harvard")
                elif stage == 'val' and row["set"] == 'val':
                    self.images.append(row["image_path"])
                    self.seg.append(row["mask_path"])
                    self.datasets.append("harvard")
                elif stage == 'test':
                    self.images.append(row["image_path"])
                    self.seg.append(row["mask_path"])
                    self.datasets.append("harvard")
        else:
            for img_path, seg_path in zip(self.image_paths[0], self.image_paths[1]):
                for seg_file in os.listdir(seg_path):
                    # if "Fluo-N3DH-CE/02" in seg_path and ("man_seg138" in seg_file or "man_seg017" in seg_file or "man_seg031" in seg_file or "man_seg187" in seg_file or "man_seg181" in seg_file):
                    #     print(f"\n***********************************************************************\nskiped: {seg_path}/{seg_file}\n***********************************************************************\n")
                    #     continue
                    # if "Fluo-N3DH-SIM+/01" in seg_path and ("/man_seg026.tif" in seg_file or "/man_seg069.tif" in seg_file):
                    #     continue
                    if seg_file.endswith(".tif"):
                        self.seg.append(os.path.join(seg_path, seg_file))
                        self.images.append(os.path.join(img_path, seg_file.replace("man_seg", "t", 1)))
                        self.datasets.append(os.path.basename(os.path.dirname(os.path.dirname(seg_path))))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # print(f"\n{img_path}")
        seg_path = self.seg[idx]
        dataset = self.datasets[idx]
        image = torch.from_numpy(tiff.imread(img_path).astype(np.float32))
        seg_mask = torch.from_numpy(tiff.imread(seg_path).astype(np.float32))

        subj = tio.Subject(image=tio.ScalarImage(tensor=image.unsqueeze(0)),
                           seg_mask=tio.LabelMap(tensor=seg_mask.unsqueeze(0)),
                           )

        subj = self.transform(subj)

        image = subj.image.tensor
        seg_mask = subj.seg_mask.tensor

        return image, seg_mask, dataset








class CELL2DDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.batch_size = data_cfg.batch_size
        self.num_workers = data_cfg.pop('num_workers')
        self.img_size = data_cfg.img_size if hasattr(data_cfg, 'img_size') else None
        self.harvard_csv = data_cfg.pop('harvard_csv') if hasattr(data_cfg, 'harvard_csv') else None

        if data_cfg.sys == 'runai':
            self.base_dir = data_cfg.pop('dir_runai')
        elif data_cfg.sys == 'dgx':
            self.base_dir = data_cfg.pop('dir_dgx')
            self.test_base_dir = data_cfg.pop('dir_dgx_test')
        elif data_cfg.sys == 'CTC':
            self.base_dir = data_cfg.pop('dir_dgx')
            self.test_base_dir = '../'
        else:
            raise ValueError("System type must be either 'runai' or 'dgx'.")
        if self.harvard_csv is None:
            self.train_dirs = [
                [os.path.join(self.base_dir, img) for img in data_cfg.train.img],
                [os.path.join(self.base_dir, seg) for seg in data_cfg.train.seg],
                # [os.path.join(self.base_dir, tra) for tra in data_cfg.train.tra]
            ]
            self.val_dirs = [
                [os.path.join(self.base_dir, img) for img in data_cfg.val.img],
                [os.path.join(self.base_dir, seg) for seg in data_cfg.val.seg],
                # [os.path.join(self.base_dir, tra) for tra in data_cfg.val.tra]
            ]
            # self.test_dirs = [
            #     [os.path.join(self.base_dir, img) for img in data_cfg.test.img],
            #     [os.path.join(self.base_dir, seg) for seg in data_cfg.test.seg],
            # ]
            self.test_dirs = [ # todo if real test
                os.path.join(self.test_base_dir, img) for img in data_cfg.test.img
            ]

        else:
            self.train_dirs = None
            self.val_dirs = None
            self.test_dirs = None
        self.transform_params = data_cfg.augmentations

    def get_transform(self, train_aug, test=False):
        def affine(image, seg_mask, p=0.5, max_degrees=25, max_scale=0.2, max_shear=10):
            if random.random() < p:
                degrees = random.uniform(-max_degrees, max_degrees)
                scale = random.uniform(-max_scale, max_scale)
                scale += 1
                shear_x = random.uniform(-max_shear, max_shear)
                shear_y = random.uniform(-max_shear, max_shear)
                aff_image = TF.affine(image, angle=degrees, translate=(0, 0), scale=scale, shear=(shear_x, shear_y))
                aff_seg_mask = TF.affine(seg_mask, angle=degrees, translate=(0, 0), scale=scale,
                                         shear=(shear_x, shear_y))

                return aff_image, aff_seg_mask
            return image, seg_mask

        def horizontal_flip(image, seg_mask, p=0.5):
            for i in range(len(image)):
                if random.random() < p:
                    image[i], seg_mask[i] = TF.hflip(image[i]), TF.hflip(seg_mask[i])
            return image, seg_mask

        def vertical_flip(image, seg_mask, p=0.5):
            for i in range(len(image)):
                if random.random() < p:
                    image[i], seg_mask[i] = TF.vflip(image[i]), TF.vflip(seg_mask[i])
            return image, seg_mask


        def random_contrast(image, p=0.5, contrast_range=(0.5, 1.5)):
            for i in range(len(image)):
                if random.random() < p:
                    contrast_factor = random.uniform(*contrast_range)
                    mean = image[i].mean()
                    image[i] = mean + contrast_factor * (image[i] - mean)
            return image

        def random_brightness(image, p=0.5, brightness_factor=(0.9, 1.1)):
            for i in range(len(image)):
                if random.random() < p:
                    brightness_adjustment = random.uniform(*brightness_factor)
                    image[i] = image[i] * brightness_adjustment
            return image
        def random_crop(image, seg_mask, dataset, crop_size, threshold=330, num_crops=8):
            iter = 0
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            if not isinstance(seg_mask, Image.Image):
                seg_mask = Image.fromarray(seg_mask)
            if crop_size is None:
                width, height = image.size
                crop_size = (height, width)

            images = np.zeros((num_crops, *crop_size), dtype=np.float32)
            masks = np.zeros((num_crops, *crop_size), dtype=np.float32)
            datasets = [dataset] * num_crops

            count = 0
            while count < num_crops:
                iter += 1
                i, j, h, w = RandomCrop.get_params(
                    image, output_size=(crop_size))
                cropped_image, cropped_seg_mask = TF.crop(image, i, j, h, w), TF.crop(seg_mask, i, j, h, w)
                cropped_image, cropped_seg_mask = np.array(cropped_image), np.array(cropped_seg_mask)

                if np.count_nonzero(cropped_seg_mask) > threshold:
                    images[count] = cropped_image
                    masks[count] = cropped_seg_mask
                    count += 1
            return images, masks, datasets

        def to_tensor(images, seg_masks):
            num_crops, image_height, image_width = images.shape
            tensor_images = torch.zeros((num_crops, 1, image_height, image_width), dtype=torch.float32)
            tensor_seg_masks = torch.zeros((num_crops, image_height, image_width), dtype=torch.float32)
            for i in range(num_crops):
                tensor_images[i][0] = torch.from_numpy(images[i])
                tensor_seg_masks[i] = torch.from_numpy(seg_masks[i])

            return tensor_images, tensor_seg_masks

        def transform(image, seg_mask, dataset):
            if train_aug:
                image, seg_mask = Image.fromarray(image), Image.fromarray(seg_mask)
                if 'affine' in self.transform_params:
                    image, seg_mask = affine(image, seg_mask, **self.transform_params.affine)
                image, seg_mask, dataset = random_crop(image, seg_mask, dataset, crop_size=self.img_size, num_crops=1)
                image = [Image.fromarray(img) for img in image]
                seg_mask = [Image.fromarray(img_seg) for img_seg in seg_mask]
                if 'horizontal_flip' in self.transform_params:
                    image, seg_mask = horizontal_flip(image, seg_mask)
                if 'vertical_flip' in self.transform_params:
                    image, seg_mask = vertical_flip(image, seg_mask)
                image = np.array([np.array(img) for img in image])
                seg_mask = np.array([np.array(img_seg) for img_seg in seg_mask])
                if 'random_contrast' in self.transform_params:
                    image = random_contrast(image, **self.transform_params.random_contrast)
                if 'random_brightness' in self.transform_params:
                    image = random_brightness(image, **self.transform_params.random_brightness)

            else:
                if test:
                    image, seg_mask, dataset = random_crop(image, seg_mask, dataset, crop_size=self.img_size, num_crops=1)
                else:
                    image, seg_mask, dataset = random_crop(image, seg_mask, dataset, crop_size=self.img_size)
            image, seg_mask = to_tensor(image, seg_mask)
            return image, seg_mask, dataset
        return transform

    def custom_collate_fn(self, batch):
        images = torch.cat([item[0] for item in batch], dim=0).to(batch[0][0])
        masks = torch.cat([item[1] for item in batch], dim=0).to(batch[0][0])
        datasets = sum([item[2] for item in batch], [])

        if len(batch[0]) == 4:  # test set with mask_name
            mask_names = [b[3] for b in batch]  # keep as list
            return images, masks, datasets, mask_names

        return images, masks, datasets

    def setup(self, stage):

        if stage == 'fit':
            transform = self.get_transform(train_aug=True)
            self.train_ds = CELL2DDataset(
                image_paths=self.train_dirs,
                transform=transform,
                harvard_csv=self.harvard_csv,
                stage='train',
            )
        if stage in (None, "fit", "validate"):
          transform = self.get_transform(train_aug=False)
          self.val_ds = CELL2DDataset(
              image_paths=self.val_dirs,
              transform=transform,
              harvard_csv=self.harvard_csv,
              stage='val',
          )
        if stage == 'test':
            transform = self.get_transform(train_aug=False, test=True)
            # self.test_ds = CELL2DDataset(
            #     # image_paths=self.val_dirs,
            #     image_paths=self.test_dirs,
            #     transform=transform,
            #     harvard_csv=self.harvard_csv,
            #     stage=stage,
            # )
            ## todo if real test
            self.test_ds = CELL2DDatasetTest(
                # image_paths=self.val_dirs,
                image_paths=self.test_dirs,
                transform=transform,
                harvard_csv=self.harvard_csv,
                stage=stage,
            )

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds, shuffle=True, batch_size=self.batch_size,
                              num_workers=self.num_workers, persistent_workers=True, drop_last=True,
                              collate_fn=self.custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds, shuffle=False, batch_size=self.batch_size,
                   num_workers=self.num_workers, persistent_workers=True, collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds, shuffle=False, batch_size=1,
                          num_workers=self.num_workers, persistent_workers=True, collate_fn=self.custom_collate_fn)

    def predict_dataloader(self):
        raise NotImplementedError

class CELL2DDataset(Dataset):
    def __init__(self, image_paths, transform, harvard_csv, stage):
        self.image_paths = image_paths
        self.images = []
        self.seg = []
        self.datasets = []
        # self.tra = []
        # for img_path, seg_path, tra_path in zip(self.image_paths[0], self.image_paths[1], self.image_paths[2]):
        #     for img in os.listdir(img_path):
        #         self.images.append(os.path.join(img_path, img))
        #         self.seg.append(os.path.join(seg_path, img.replace("t", "man_seg", 1)))
        #         self.tra.append(os.path.join(tra_path, img.replace("t", "man_track", 1)))
        if harvard_csv:
            df = pd.read_csv(harvard_csv)
            for _, row in df.iterrows():

                if stage == 'train' and row["set"] == 'train':
                    self.images.append(row["image_path"])
                    self.seg.append(row["mask_path"])
                    self.datasets.append("harvard")
                elif stage == 'val' and row["set"] == 'val':
                    self.images.append(row["image_path"])
                    self.seg.append(row["mask_path"])
                    self.datasets.append("harvard")
                elif stage == 'test':
                    self.images.append(row["image_path"])
                    self.seg.append(row["mask_path"])
                    self.datasets.append("harvard")
        else:
            for img_path, seg_path in zip(self.image_paths[0], self.image_paths[1]):
                # for img in os.listdir(img_path):
                #     self.images.append(os.path.join(img_path, img))
                #     self.seg.append(os.path.join(seg_path, img.replace("t", "man_seg", 1)))
                for seg_file in os.listdir(seg_path):
                    if seg_file.endswith(".tif"):
                        self.seg.append(os.path.join(seg_path, seg_file))
                        self.images.append(os.path.join(img_path, seg_file.replace("man_seg", "t", 1)))
                        self.datasets.append(os.path.basename(os.path.dirname(os.path.dirname(seg_path))))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        seg_path = self.seg[idx]
        dataset = self.datasets[idx]
        # tra_path = self.tra[idx]

        image = tiff.imread(img_path).astype(np.float32)
        seg_mask = tiff.imread(seg_path).astype(np.float32)
        # tra_mask = tiff.imread(tra_path).astype(np.float32)

        image = (image - image.mean()) / (image.std())

        # image, seg_mask, tra_mask = self.transform(image=image, seg_mask=seg_mask, tra_mask=tra_mask)
        image, seg_mask, dataset = self.transform(image=image, seg_mask=seg_mask, dataset=dataset)

        return image, seg_mask, dataset

class CELL2DDatasetTest(Dataset):
    def __init__(self, image_paths, transform, harvard_csv, stage):
        self.image_paths = image_paths
        self.images = []
        self.datasets = []
        # self.tra = []
        # for img_path, seg_path, tra_path in zip(self.image_paths[0], self.image_paths[1], self.image_paths[2]):
        #     for img in os.listdir(img_path):
        #         self.images.append(os.path.join(img_path, img))
        #         self.seg.append(os.path.join(seg_path, img.replace("t", "man_seg", 1)))
        #         self.tra.append(os.path.join(tra_path, img.replace("t", "man_track", 1)))
        if harvard_csv:
            df = pd.read_csv(harvard_csv)
            for _, row in df.iterrows():
                self.images.append(row["image_path"])
                self.seg.append(row["mask_path"])
                self.datasets.append("harvard")
        else:
            for img_path in self.image_paths:
                # for img in os.listdir(img_path):
                #     self.images.append(os.path.join(img_path, img))
                #     self.seg.append(os.path.join(seg_path, img.replace("t", "man_seg", 1)))
                for img_file in os.listdir(img_path):
                    if img_file.endswith(".tif"):
                        self.images.append(os.path.join(img_path, img_file))
                        self.datasets.append(os.path.basename(os.path.dirname(img_path)))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        dataset = self.datasets[idx]
        # tra_path = self.tra[idx]

        image = tiff.imread(img_path).astype(np.float32)

        image = (image - image.mean()) / (image.std())

        # image, seg_mask, tra_mask = self.transform(image=image, seg_mask=seg_mask, tra_mask=tra_mask)
        image, seg_mask, dataset = self.transform(image=image, seg_mask=image, dataset=dataset)

        p = Path(img_path)
#        mask_name = str(Path(p.parent.name) / p.name.replace("t", "mask", 1))
        mask_name = str(p.name.replace("t", "mask", 1))
        return image, seg_mask, dataset, mask_name

