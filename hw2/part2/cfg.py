

## You could add some configs to perform other training experiments...

LeNet_cfg = {
    'model_type': 'LeNet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 512,
    'lr':0.001,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 100,
    
}

myResnet_cfg = {
    'model_type': 'myResnet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    'split_ratio': 0.9,
    'seed': 678,
    
    # training hyperparameters
    'batch_size': 512,
    'lr':0.001,
    'milestones': [25, 50],
    'num_out': 10,
    'num_epoch': 100,
    
}

Resnet50_cfg = {
    'model_type': 'Resnet50',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    'split_ratio': 0.9,
    'seed': 678,
    
    # training hyperparameters
    'batch_size': 512,
    'lr':0.001,
    'milestones': [25, 50],
    'num_out': 10,
    'num_epoch': 100,
    
}