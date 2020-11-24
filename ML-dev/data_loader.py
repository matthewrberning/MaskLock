from pathlib import Path
import numpy as np

# import dlib
# import MTCNN

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN


from utils import load_image_and_preprocess


def collate_fn(batch):
    imgs = [item['image'] for item in batch if item['image'] is not None]
    targets = [item['label'] for item in batch if item['image'] is not None]
    filenames = [item['filename'] for item in batch if item['image'] is not None]
    imgs = torch.stack(imgs)
    targets = torch.stack(targets)
    return {'image': imgs, 'label': targets, 'filename': filenames}


def get_transforms():
    pre_trained_mean, pre_trained_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.02, 0.16), ratio=(0.3, 1.6)),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std),
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])
    return train_transforms, val_transforms


class MaskDataset(Dataset):
    def __init__(self, filenames, transform, output_image_size=224):
        self.filenames = filenames
        self.transform = transform
        self.image_size = output_image_size
        # self.face_detector = dlib.get_frontal_face_detector()
         # three steps's threshold
        self.face_detector = MTCNN(keep_all=True,  thresholds=[0.57, 0.68, 0.68])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image_id = filename.stem
        filename = str(filename)

        #set labes for images mask vs. no-mask
        label = 1 if 'mask' in filename.split('/') else 0
        # print(str(filename) + str(self.image_size) + str(self.face_detector))
        image = load_image_and_preprocess(filename, self.image_size, self.face_detector)
        if image is None:
            # print("\n\nNONE", filename)
            image = []
    
        
        if len(image) == 0:
            return {'image': None, 'label': None ,'filename': filename}
        
        image = Image.fromarray(image)
        image = self.transform(image)
        label = torch.tensor(label)
        
        return {'image': image, 'label': label, 'filename': filename}
    

def create_dataloaders(dataset_path, batch_size):
    train_transforms, val_transforms = get_transforms()

    train_dl = _create_dataloader(f"{dataset_path}train/", batch_size=batch_size, transformations=train_transforms, mode="train")

    val_dl = _create_dataloader(f"{dataset_path}val/", batch_size=batch_size, transformations=val_transforms, mode="val")

    return train_dl, val_dl


def _create_dataloader(file_paths, batch_size, transformations, mode):

    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    
    filenames = []
    for file_path in file_paths:
        data_path = Path(file_path)
    
        mask_filenames = _find_filenames(data_path / 'mask/', '*.png')
        no_mask_filenames = _find_filenames(data_path / 'no_mask/', '*.png')
        
        filenames += mask_filenames
        filenames += no_mask_filenames
        
    assert len(filenames) != 0, f'filenames are empty {filenames}'
    np.random.shuffle(filenames)
    
    ds = MaskDataset(filenames, transform=transformations)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    print(f"{mode} data: {len(ds)}")
    
    return dl


def _find_filenames(file_dir_path, file_pattern): return list(file_dir_path.glob(file_pattern))

