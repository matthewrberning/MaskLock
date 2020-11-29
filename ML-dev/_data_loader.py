from pathlib import Path
import numpy as np

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler

import cv2

def collate_fn(batch):
    imgs = [item['image'] for item in batch if item['image'] is not None]
    targets = [item['label'] for item in batch if item['image'] is not None]
    filenames = [item['filename'] for item in batch if item['image'] is not None]
    imgs = torch.stack(imgs)
    targets = torch.stack(targets)
    return {'image': imgs, 'label': targets, 'filename': filenames}

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
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image_id = filename.stem
        filename = str(filename)

        #set labes for images mask vs. no-mask based upon directory affiliation
        # label = 1 if 'mask' in filename.split('/') else 0
        #///////////////////////////////////////////////////////////////////////////label upsampling
        # label = 1 if 'up_mask' in filename.split('/') else 0        
        #this ^^^ stupid thing was writen with the wrong slash / not \ for windows? ABSOLOUTELY FRUSTRATING
        label = 1 if 'up_mask' in filename.split('\\') else 0

        image = cv2.imread(filename)
        image = Image.fromarray(image)
        image = self.transform(image)
        label = torch.tensor(label)
        
        return {'image': image, 'label': label, 'filename': filename}
    

def build_dataloaders(dataset_path, batch_size):

    #get the composed transforms for training and validation(just normalization)
    t_transforms, v_transforms = compose_transforms()

    #collect the files from the training set and send to dataloader/MaskDataset class
    train_filenames = get_files(f"{dataset_path}train/")
    trainset_class = MaskDataset(train_filenames, transform=t_transforms)
    train_dl = DataLoader(trainset_class, batch_size=batch_size, sampler=RandomSampler(trainset_class), collate_fn=collate_fn)
    
    print("train data: ", len(trainset_class))
    
    #collect the files from the validation set and send to dataloader/MaskDataset class
    val_filenames = get_files(f"{dataset_path}val/")
    valset_class = MaskDataset(val_filenames, transform=v_transforms)
    val_dl = DataLoader(valset_class, batch_size=batch_size, sampler=RandomSampler(valset_class), collate_fn=collate_fn)
    
    print("val data: ", len(valset_class))
    
    return train_dl, val_dl

def get_files(dataset_path):
    files = [] #empty list to accumulate all the image_paths
 
    data_path = Path(dataset_path) #pathify it

    #///////////////////////////////////////////////////////////////////////////label upsampling
    mask_images =  collect_files(data_path / 'up_mask/', '*.png') #use helper fn
    # mask_images =  collect_files(data_path / 'mask/', '*.png') #use helper fn
    no_mask_images = collect_files(data_path / 'no_mask/', '*.png') #use helper fn

    
    files += mask_images
    files += no_mask_images
        
    assert len(files) != 0, f'No images were collected! Check this path: {dataset_path}'
    
    np.random.shuffle(files)

    return files

def collect_files(file_dir_path, file_pattern): 
    return list(file_dir_path.glob(file_pattern))