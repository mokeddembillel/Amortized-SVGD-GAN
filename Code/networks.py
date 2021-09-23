import seaborn as sns
sns.set()

import torch as T
import torch.nn as nn
import torch.nn.functional as F



class G(nn.Module):
    def __init__(self,lr=1e-4, input_dim=2):
        super().__init__()
        
        # Initialize Input dimentions
        self.fc1_dim = 2
        self.fc2_dim = 128
        self.fc3_dim = 50
        self.fc4_dim = 50
        self.fc5_dim = 2
        
        
        
        self.fc1 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc2 = nn.Linear(self.fc2_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.fc2_dim)
        self.fc4 = nn.Linear(self.fc2_dim, self.fc2_dim)
        self.fc5 = nn.Linear(self.fc2_dim, self.fc5_dim)
        
        
        self.bn1 = nn.BatchNorm1d(self.fc2_dim)
        self.bn2 = nn.BatchNorm1d(self.fc2_dim)
        self.bn3 = nn.BatchNorm1d(self.fc2_dim)
        self.bn4 = nn.BatchNorm1d(self.fc2_dim)
        

        # Initialize layers weights
        self.fc1.weight.data.normal_(0,1)
        self.fc2.weight.data.normal_(0,1)
        self.fc3.weight.data.normal_(0,1)
        self.fc4.weight.data.normal_(0,1)
        self.fc5.weight.data.normal_(0,1)
        
    
        # Initialize layers biases
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.constant_(self.fc4.bias, 0)
        nn.init.constant_(self.fc5.bias, 0)
        
        
        self.lr = lr
        # Define Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr = self.lr, betas=(0.0, 0.99))
        
        # Set Device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, input, num_particle=5):
        X = self.fc1(input)
        X = F.tanh(self.bn1(X))
        X = self.fc2(X)
        X = F.tanh(self.bn2(X))
        X = self.fc3(X)
        X = F.tanh(self.bn3(X))
        X = self.fc4(X)
        X = F.tanh(self.bn4(X))
        X = self.fc5(X)
        return X
    



# In[ ]:
    

class D(nn.Module):
    def __init__(self,lr=1e-4, input_dim=2):
        super().__init__()
        
        # Initialize Input dimentions
        self.fc1_dim = 2
        self.fc2_dim = 128
        self.fc3_dim = 1
        
        #self.fc1 = nn.Linear(self.fc1_dim, self.fc3_dim)
        
        # Define the NN layers
        self.fc1 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc2 = nn.Linear(self.fc2_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.fc3_dim)
        
        # self.bn1 = nn.BatchNorm1d(self.fc2_dim)
        # self.bn2 = nn.BatchNorm1d(self.fc2_dim)
        
        # Initialize layers weights
        self.fc1.weight.data.normal_(0,0.2)
        self.fc2.weight.data.normal_(0,0.2)
        self.fc3.weight.data.normal_(0,0.2)
        
        # # Initialize layers biases
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)
    
        self.lr = lr
        # Define Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr = self.lr, betas=(0.0, 0.99))
        
        
        # Set Device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, input):
        
        X = F.relu(self.fc1(input))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return -T.log(F.sigmoid(X))
        
