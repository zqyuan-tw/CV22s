

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import os
import numpy as np 
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.transforms import transforms
from PIL import Image
import json
from tqdm import tqdm



def get_cifar10_train_val_set(root, ratio=0.9, cv=0):
    
    # get all the images path and the corresponding labels
    with open(root, 'r') as f:
        data = json.load(f)
    images, labels = data['images'], data['categories']

    
    info = np.stack( (np.array(images), np.array(labels)) ,axis=1)
    N = info.shape[0]

    # apply shuffle to generate random results 
    np.random.shuffle(info)
    x = int(N*ratio) 
    
    all_images, all_labels = info[:,0].tolist(), info[:,1].astype(np.int32).tolist()


    train_image = all_images[:x]
    val_image = all_images[x:]

    train_label = all_labels[:x] 
    val_label = all_labels[x:]
    

    
    ## TO DO ## 
    # Define your own transform here 
    # It can strongly help you to perform data augmentation and gain performance
    # ref: https://pytorch.org/vision/stable/transforms.html
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
                ## TO DO ##
                # You can add some transforms here
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                
                # ToTensor is needed to convert the type, PIL IMG,  to the typ, float tensor.  
                transforms.ToTensor(),
                
                # experimental normalization for image classification 
                transforms.Normalize(means, stds),
            ])
  
    # normally, we dont apply transform to test_set or val_set
    val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])

 
  
    ## TO DO ##
    # Complete class cifiar10_dataset
    train_set, val_set = cifar10_dataset(images=train_image, labels=train_label,transform=train_transform), \
                        cifar10_dataset(images=val_image, labels=val_label,transform=val_transform)


    return train_set, val_set

## TO DO ##
# Define your own cifar_10 dataset
class cifar10_dataset(Dataset):
    def __init__(self,images , labels=None , transform=None, prefix = './p2_data/train'):
        
        # It loads all the images' file name and correspoding labels here
        self.images = images 
        self.labels = labels 
        
        # The transform for the image
        self.transform = transform
        
        # prefix of the files' names
        self.prefix = prefix
        
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        ## TO DO ##
        # You should read the image according to the file path and apply transform to the images
        # Use "PIL.Image.open" to read image and apply transform
        
        # You shall return image, label with type "long tensor" if it's training set
        return (self.transform(Image.open(os.path.join(self.prefix, self.images[idx])).convert("RGB")), self.labels[idx]) if self.labels is not None else self.transform(Image.open(os.path.join(self.prefix, self.images[idx])).convert("RGB"))
        
def data_cleaning(dataset, model, threshold, device, ratio, n_epoch, batch_size, model_type):
    os.makedirs( os.path.join('./dirty_img', model_type), exist_ok=True)
    dirty_path = os.path.join('./dirty_img', model_type, model_type + '.png')
    os.makedirs( os.path.join('./weak_ckpt',  model_type), exist_ok=True)
    weak_ckpt = os.path.join('./weak_ckpt', model_type, model_type + '.ckpt')
    print(f"There are {len(dataset)} images in total.")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer_weak = optim.Adam(model.parameters(), lr=0.0001)
    criterion_weak = nn.CrossEntropyLoss()
    print(f"Training a weak classifier for {n_epoch} epoch.")
    model.train()
    for epoch in range(n_epoch):
        corr_num = 0
        for data, label in tqdm(data_loader):
            data, label = data.to(device), label.to(device)
            output = model(data) 
            loss = criterion_weak(output, label)
            optimizer_weak.zero_grad()
            loss.backward()
            optimizer_weak.step()
            pred = output.argmax(dim=1)
            corr_num += (pred.eq(label.view_as(pred)).sum().item())
        print(f"epoch {epoch + 1} weak classifier accuracy {corr_num / len(data_loader.dataset)}")
    torch.save(model.state_dict(), weak_ckpt)
    # get clean set
    means = torch.tensor([0.485, 0.456, 0.406])
    stds = torch.tensor([0.229, 0.224, 0.225])
    clean_set = []
    dirty_set = []
    with torch.no_grad():
        model.eval()
        for data, label in data_loader:
            prob, idx = F.softmax(model(data.to(device)), dim=1).max(dim=1)
            for i in range(data.shape[0]):
                if prob[i] >= threshold and idx[i] == label[i]:
                    clean_set.append((data[i], label[i]))
                else:
                    if len(dirty_set) < 100:
                        dirty_set.append(data[i] * stds[:, None, None] + means[:, None, None])
    torchvision.utils.save_image(dirty_set, dirty_path, nrow=10)
    print(f"There are {len(clean_set)} images left after data cleaning. {len(dataset) - len(clean_set)} images are considered as dirty data.")
    train_set_size = int(len(clean_set) * ratio)
    val_set_size = len(clean_set) - train_set_size
    train_set, val_set = random_split(clean_set, [train_set_size, val_set_size])
    return train_set, val_set