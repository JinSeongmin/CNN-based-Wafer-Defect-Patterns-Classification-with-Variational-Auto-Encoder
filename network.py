import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(28*28*3, 128, bias=True),
                                 nn.ReLU())
        
        self.layer2 = nn.Sequential(nn.Linear(128, 128, bias=True),
                                 nn.ReLU())
        
        self.layer3 = nn.Linear(128, 9, bias=True)
        
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias, 0)
        
        
        self.layer_list = [self.layer1, self.layer2, self.layer3]
        self.layers = nn.ModuleList(self.layer_list)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i in range(len(self.layers)): 
            x = self.layers[i](x)
            
        return x



class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.layer1 =nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))

    
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2), 
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Linear(7*7*128, 9, bias=True)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias, 0)
        
        
        self.layer_list = [self.layer1, self.layer2, self.layer3]
        self.layers = nn.ModuleList(self.layer_list)


    def forward(self, x):
        for i in range(len(self.layers)): 
            if i == 2 :         
                x = x.view(x.size(0), -1)
            x = self.layers[i](x)
            
        return x



class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.layer1 =nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=True),
                                  nn.ReLU())

        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
    
        self.layer4 = nn.Sequential(nn.Linear(7*7*256, 256, bias=True),
                                 nn.ReLU())

        self.layer5 = nn.Linear(256, 9, bias=True)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias, 0)
            
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias, 0)
        
        
        self.layer_list = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
        self.layers = nn.ModuleList(self.layer_list)
        
        
    def forward(self, x):
        for i in range(len(self.layers)): 
            if i == 3 :         
                x = x.view(x.size(0), -1)
            x = self.layers[i](x)
            
        return x

