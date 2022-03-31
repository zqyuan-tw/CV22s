
# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

    
class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channels))

        self.relu = nn.ReLU()
        
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        y = self.relu(self.conv1(x))
        x = self.relu(self.conv2(y) + x)
        return x

        
class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()
        
        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        
        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.
        
        def down_sample(cin, cout):
            return nn.Sequential(
                nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(cout),
                nn.ReLU()
            )

        self.res_block1 = residual_block(64)
        self.res_block2 = residual_block(128)
        self.res_block3 = residual_block(256)

        self.down_sample1 = down_sample(64, 128)
        self.down_sample2 = down_sample(128, 256)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

        self.fc = nn.Linear(in_features=4096, out_features=num_out)
        
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        # x = x.flatten(x)
        # print(x.shape)

        x = self.relu(self.stem_conv(x)) # 32

        x = self.down_sample1(self.res_block1(x)) # 16
        x = self.down_sample2(self.res_block2(x)) # 8
        x = self.res_block3(x)

        x = self.maxpool(x) # 4

        x = self.fc(x.view(x.size(0), -1))
        
        return x

if __name__ == '__main__':
    input = torch.rand(2, 3, 32, 32)
    net = myResnet()
    output = net(input)
    print(output.shape) 
