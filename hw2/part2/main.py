
from copy import deepcopy
import torch
import os

from torchvision.models import resnet18, resnet50

import torch.optim as optim 
import torch.nn as nn

from myModels import  myLeNet, myResnet
from myDatasets import  get_cifar10_train_val_set, data_cleaning
from tool import train, fixed_seed

# Modify config if you are conducting different models
# from cfg import LeNet_cfg as cfg
# from cfg import Resnet_cfg as cfg
from cfg import Resnet50_cfg as cfg


def train_interface():
    
    """ input argumnet """

    data_root = cfg['data_root']
    model_type = cfg['model_type']
    num_out = cfg['num_out']
    num_epoch = cfg['num_epoch']
    split_ratio = cfg['split_ratio']
    seed = cfg['seed']
    
    # fixed random seed
    fixed_seed(seed)
    

    os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True) 
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)


    # with open(log_path, 'w'):
    #     pass
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu') 
    
    
    """ training hyperparameter """
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    milestones = cfg['milestones']
    
    
    ## Modify here if you want to change your model ##
    # model = myLeNet(num_out=num_out)
    # model = myResnet(num_out=num_out)
    model = resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_out)
    )
    # print model's architecture
    print(model)

    # Get your training Data 
    ## TO DO ##
    # You need to define your cifar10_dataset yourself to get images and labels for earch data
    # Check myDatasets.py 
      
    train_set, val_set =  get_cifar10_train_val_set(root=data_root, ratio=split_ratio)

    # train a weak classifier
    # all_set, _ = get_cifar10_train_val_set(data_root, ratio=1, cv=0)
    # train_set, val_set = data_cleaning(all_set, deepcopy(model), 0.1, device, split_ratio, 50, batch_size, model_type)
    
    # define your loss function and optimizer to unpdate the model's parameters.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=10, cooldown=3)
    
    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    criterion = nn.CrossEntropyLoss()
    
    # Put model's parameters on your device
    model = model.to(device)
    
    ### TO DO ### 
    # Complete the function train
    # Check tool.py
    train(model=model, train_set=train_set, val_set=val_set, batch_size=batch_size,
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler, do_semi=True, unlabel_dir='./p2_data/unlabeled')

    
if __name__ == '__main__':
    train_interface()

