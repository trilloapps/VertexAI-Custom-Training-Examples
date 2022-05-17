import torch
import torchvision
from torchvision.transforms import transforms
import cv2
from google.cloud import storage
import os

from torch.utils.data import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

class ImageDataset(Dataset):
    
    def __init__(self, dataframe, is_train, is_test):

        self.is_train = is_train
        self.is_test = is_test
        
        self.image_names = list(dataframe['Id'].values)
        self.labels = list(dataframe.drop('Id', axis=1).values)
        
        # Set the traning transform
        if self.is_train == True:
            print(f'Number of training images: {len(dataframe)}')
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
                transforms.Normalize([0.4321, 0.3840, 0.3556], [0.3130, 0.2917, 0.2786])
            ])
        
        # Set the validation transform
        elif self.is_train == False and self.is_test == False:
            print(f'Number of validation images: {len(dataframe)}')
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.4321, 0.3840, 0.3556], [0.3130, 0.2917, 0.2786])
            ])
        
        # Set the Testing transform
        elif self.is_train == False and self.is_test == True:
            print(f'Number of testing images: {len(dataframe)}')
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.4321, 0.3840, 0.3556], [0.3130, 0.2917, 0.2786])
            ])
                       
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):

        image = cv2.imread(f'./images/{self.image_names[index]}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        targets = self.labels[index]
        
        image_detach = image.detach().clone()
        
        return image_detach.type(torch.float32), torch.tensor(targets, dtype=torch.float32)
