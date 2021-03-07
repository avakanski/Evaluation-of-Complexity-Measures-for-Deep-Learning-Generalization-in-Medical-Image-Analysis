# This file is used for loading the dataset

import os
from os import listdir
from torch.utils.data.dataset import Dataset
from PIL import Image
import natsort 
import pandas


# Dataset class for classification task
class SingleTask_DataSet(Dataset):
    def __init__(self, imgs_dir, transform, labels_file):
        # directory with images
        self.imgs_dir = imgs_dir
        self.transform = transform
        # labels file
        self.labels_file = labels_file

        all_imgs = os.listdir(imgs_dir)
        # sort the image files in ascending order
        self.total_imgs = natsort.natsorted(all_imgs)
        
        # read the fabels file
        labels_pd = pandas.read_excel(labels_file)
        labels_tumors = labels_pd['Type'].values
        # assign 1s to malignant tumors, 0s to benign tumors
        self.all_labels = (labels_tumors == 'Malignant').astype(int)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.imgs_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_image = self.transform(image)
        label = self.all_labels[idx]
        return (tensor_image, label)

# Dataset class for joint classification and segmentation task        
class MultiTask_DataSet(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform, labels_file):
        # directory with images
        self.imgs_dir = imgs_dir
        # directory with segmentation masks
        self.masks_dir = masks_dir
        self.transform = transform
        # labels file
        self.labels_file = labels_file
        
        all_imgs = os.listdir(imgs_dir)
        # sort the image files in ascending order
        self.total_imgs = natsort.natsorted(all_imgs)

        all_masks = os.listdir(masks_dir)
        # sort the mask files in ascending order
        self.total_masks = natsort.natsorted(all_masks)

        # read the fabels file
        labels_pd = pandas.read_excel(labels_file)
        labels_tumors = labels_pd['Type'].values
        # assign 1s to malignant tumors, 0s to benign tumors
        self.all_labels = (labels_tumors == 'Malignant').astype(int)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.imgs_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_image = self.transform(image)
        mask_loc = os.path.join(self.masks_dir, self.total_masks[idx])
        mask = Image.open(mask_loc).convert('L')
        tensor_mask = self.transform(mask)
        label = self.all_labels[idx]
        return (tensor_image, tensor_mask, label)


