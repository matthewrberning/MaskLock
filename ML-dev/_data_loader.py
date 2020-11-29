from pathlib import Path
import numpy as np

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

import cv2


def compose_transforms():
    pre_trained_mean, pre_trained_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
        transforms.RandomPerspective(distortion_scale=0.2),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        # transforms.RandomErasing(scale=(0.02, 0.16), ratio=(0.3, 1.6)),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std),
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])
    return train_transforms, val_transforms


class MaskDataset(Dataset):
    def __init__(self, filenames, transform):
        self.filenames = filenames
        self.transform = transform
        self.image_size = output_image_size
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image_id = filename.stem
        filename = str(filename)

        #set labes for images mask vs. no-mask based upon directory affiliation
        label = 1 if 'mask' in filename.split('/') else 0

        image = cv2.imread(filename)
        image = Image.fromarray(image)
        image = self.transform(image)
        label = torch.tensor(label)
        
        return {'image': image, 'label': label, 'filename': filename}
    

def build_dataloaders(dataset_path, batch_size, device):

    #get the composed transforms for training and validation(just normalization)
    t_transforms, v_transforms = compose_transforms()

    #collect the files from the training set and send to dataloader/MaskDataset class
    train_filenames = get_files(f"{dataset_path}train/")
    train_dl = DataLoader(MaskDataset(train_filenames, transform=t_transforms), batch_size=batch_size, shuffle=True)

    #collect the files from the validation set and send to dataloader/MaskDataset class
    val_filenames = get_files(f"{dataset_path}val/")
    val_dl = DataLoader(MaskDataset(val_filenames, transform=v_transforms), batch_size=batch_size, shuffle=True)

    return train_dl, val_dl


def get_files(dataset_path)
    files = [] #empty list to accumulate all the image_paths
 
    data_path = Path(dataset_path) #pathify it

    mask_images =  collect_files(data_path / 'mask/', '*.png') #use helper fn
    no_mask_images = collect_files(data_path / 'no_mask/', '*.png') #use helper fn
    
    files += mask_images
    files += no_mask_images
        
    assert len(files) != 0, f'No images were collected! Check this path: {dataset_path}'
    
    np.random.shuffle(files)

    return files

def collect_files(file_dir_path, file_pattern): 
    return list(file_dir_path.glob(file_pattern))