from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    Rand3DElasticd,
    RandShiftIntensityd,
    RandGaussianNoised,
    EnsureTyped,
    Compose,
)
import numpy as np
from monai.data import SmartCacheDataset
from torch.utils.data import DataLoader


def train_transforms(image, label):
    transform = Compose([
        LoadImaged(keys=['image','label']),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(2, 1.62, 1.62), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-80,
            a_max=305,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(160, 160, 64),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        Rand3DElasticd(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.5,
            sigma_range=(5, 8),
            magnitude_range=(50, 150),
            spatial_size=(160, 160, 64),
            translate_range=(10, 10, 5),
            rotate_range=(np.pi/36, np.pi/36, np.pi),
            scale_range=(0.1, 0.1, 0.1),
            padding_mode="zeros",
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.25,
        ),
        RandGaussianNoised(keys=["image"], prob=0.25, mean=0.0, std=0.1),
        EnsureTyped(keys=["image", "label"]),
    ])


def test_transforms(image, label):
    transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(2, 1.62, 1.62), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-80,
            a_max=305,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(150, 150, 64),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        EnsureTyped(keys=["image", "label"]),
    ])

print("CREATING TRAIN DS", flush=True)


def val_transforms(image, label):
    transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(2, 1.62, 1.62), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-80,
            a_max=305,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])

#CREATING TRAINING DATASET
def create_train_dataset(train_files, train_transforms):
    train_ds = SmartCacheDataset(data=train_files, transform=train_transforms, cache_rate=0.1, replace_rate=0.5)
    print(len(train_ds))
    print("CREATED TRAIN DS", flush=True)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    print("CREATED TRAIN DATALOADER", flush=True)
    return train_loader 

#CREATING VALIDATION DATASET
def create_val_dataset(val_files):
    val_ds = SmartCacheDataset(data=val_files, transform=val_transforms, cache_rate=0.1,replace_rate=0.5)
    print("CREATED VAL DS", flush=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)
    print("CREATED VAL DATALOADER", flush=True)

#CREATING TESTING DATASET
def create_test_dataset(test_files):
    test_ds = SmartCacheDataset(data=test_files, transform=test_transforms, cache_rate=0.1,replace_rate=0.5)
    print("CREATED TEST DS", flush=True)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)
    print("CREATED TEST DATALOADER", flush=True)