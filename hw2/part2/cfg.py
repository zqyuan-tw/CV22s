

## You could add some configs to perform other training experiments...

LeNet_cfg = {
    'model_type': 'LeNet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 128,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 30,
    
}

Resnet_cfg = {
    'model_type': 'Resnet',
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