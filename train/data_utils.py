import os
import pandas as pd
from torch.utils.data import Dataset
import boto3
import numpy as np
import torch
import io
class CIFAR10_dataset(Dataset):
    def __init__(self, train, n_components):
        bucket = 'pca-images'
        s3_client = boto3.client('s3')
        
        if(train == True):
            key_images = 'PCA_images/pca_train_images_{}x{}.pt'.format(np.sqrt(n_components).astype('uint8'), np.sqrt(n_components).astype('uint8'))
            key_labels = 'PCA_images/pca_train_labels.pt'
        else:
            key_images = 'PCA_images/pca_test_images_{}x{}.pt'.format(np.sqrt(n_components).astype('uint8'), np.sqrt(n_components).astype('uint8'))
            key_labels = 'PCA_images/pca_test_labels.pt'   
                                                   
        s3_client = boto3.client('s3')
        obj = s3_client.get_object(Bucket = bucket, Key = key_images)
        labels = s3_client.get_object(Bucket = bucket, Key = key_labels)
                                                       
                                                       
        self.images = torch.load(io.BytesIO(obj['Body'].read()))
        self.labels = torch.load(io.BytesIO(labels['Body'].read()))
        self.length = len(self.labels)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.images[idx].permute(2,0,1)
        label = self.labels[idx]
        return image, label