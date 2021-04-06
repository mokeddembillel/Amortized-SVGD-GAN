# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch as T
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F


# In[ ]:


import torch 
print(torch.__version__)


# # Make data

# In[ ]:


f = lambda x : x**2

x = np.linspace(-3,3,300)
mean_list = f(x)
#y = f(x) + np.random.uniform(low=-1.0,high=1.0,size=x.shape)* 4* np.cos(x/2+.5)
y = f(x) + np.random.normal(0,1,size=x.shape) + np.random.normal(1,1,size=x.shape)

y_ = -y
x = np.concatenate((x,x), axis=0)
y = np.concatenate((y,y_),axis =0)
len(x)
len(y)


# In[ ]:


plt.figure(figsize=(10,8))
plt.xlim(-10,10)
plt.ylim(-30,30)
plt.scatter(x, y,s=3)


# # Define net

# In[ ]:


class G(nn.Module):
    def __init__(self,lr=1e-3, input_dim=2):
        super().__init__()
        
        # Initialize Input dimentions
        self.input_dim = 2
        self.input_fc1_dim = 64
        self.fc1_output_dim = 64
        self.output_dim = 2
        
        # Define the NN layers
        self.input_layer = nn.Linear(self.input_dim, self.input_fc1_dim)
        self.fc1 = nn.Linear(self.input_fc1_dim, self.fc1_output_dim)
        self.output_layer = nn.Linear(self.fc1_output_dim, self.output_dim)
        
        # Initialize layers weights
        self.input_layer.weight.data.normal_(0,1)
        self.fc1.weight.data.normal_(0,1)
        self.output_layer.weight.data.normal_(0,1)
        
        # Initialize layers biases
        nn.init.constant_(self.input_layer.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)
        
        self.lr = lr
        # Define Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr = self.lr)
        
        # Set Device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, input, num_particle=5):
        X = self.input_layer(input)
        X = F.relu(X)
        X = self.fc1(X)
        X = F.relu(X)
        output = self.output_layer(X)
        #output = F.relu(output)
        return output
    



# In[ ]:
    
class D(nn.Module):
    def __init__(self,lr=1e-4, input_dim=2):
        super().__init__()
        
        # Initialize Input dimentions
        self.input_dim = 4
        self.input_fc1_dim = 64
        self.fc1_output_dim = 64
        self.output_dim = 1
        
        # Define the NN layers
        self.input_layer = nn.Linear(self.input_dim, self.input_fc1_dim)
        self.fc1 = nn.Linear(self.input_fc1_dim, self.fc1_output_dim)
        self.output_layer = nn.Linear(self.fc1_output_dim, self.output_dim)
        
        # Initialize layers weights
        self.input_layer.weight.data.normal_(0,1)
        self.fc1.weight.data.normal_(0,1)
        self.output_layer.weight.data.normal_(0,1)
        
        # Initialize layers biases
        nn.init.constant_(self.input_layer.bias, 0.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)
        
        self.lr = lr
        # Define Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr = self.lr)
        
        # Set Device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, f_x, x_obs):
        input = T.cat((f_x, x_obs), dim=1)
        print(input)
        X = self.input_layer(input)
        X = F.relu(X)
        X = self.fc1(X)
        X = F.relu(X)
        output = self.output_layer(X)
        return output


# # Calculate the amortized Stein Variational Gradient
# 
# $$
# \begin{eqnarray}
# \eta^{t+1} &\leftarrow \eta^{t} + \epsilon \frac{1}{m}\sum_{i=1}^m \partial_{\eta}f(\xi_i ; \eta^t) \phi^* (z_i)\\
# &\leftarrow \eta^{t} + \epsilon \frac{1}{nm} \sum_{i=1}^m \partial_{\eta}f(\xi_i ; \eta^t) \sum_{j=1}^{n} \big[ \nabla_{z_j} \log p(z_j) k(z_j, z_i) + \nabla_{z_j}k(z_j,z_i) \big]\\
# \end{eqnarray}
# $$

# In[ ]:


def rbf_kernel(input_1, input_2,  h_min=1e-3):
    
    k_fix, out_dim1 = input_1.size()[-2:]
    k_upd, out_dim2 = input_2.size()[-2:]
    assert out_dim1 == out_dim2

    leading_shape = input_1.size()[:-2]
    # Compute the pairwise distances of left and right particles.
    diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)  
    # N * k_fix * 1 * out_dim / N * 1 * k_upd * out_dim/ N * k_fix * k_upd * out_dim
    dist_sq = diff.pow(2).sum(-1)
    # N * k_fix * k_upd
    dist_sq = dist_sq.unsqueeze(-1)
    # N * k_fix * k_upd * 1
    
    # Get median.
    median_sq = torch.median(dist_sq, dim=1)[0]
    median_sq = median_sq.unsqueeze(1)
    # N * 1 * k_upd * 1
    
    h = median_sq / np.log(k_fix + 1.)
    # N * 1 * k_upd * 1

    kappa = torch.exp(- dist_sq / h)
    # N * k_fix * k_upd * 1

    # Construct the gradient
    kappa_grad = -2. * diff / h * kappa
    return kappa, kappa_grad

def learn_G(g_net, d_net, x_obs, batch_size = 10, alpha=1.):
    # Draw zeta random samples
    zeta = T.FloatTensor(batch_size, 2).uniform_(0, 1)
    # Forward the noise through the network
    f_x = g_net.forward(zeta)
    ### Equation 7 (Compute Delta xi)
    # Get the energy using the discriminator
    score = d_net.forward(f_x, x_obs)
    # Get the Gradients of the energy with respect to x and y
    grad_score = autograd.grad(score.sum(), f_x)[0].squeeze(-1)
    # grad_score_x = grad_score[:, 0]
    # grad_score_y = grad_score[:, 1]
    # grad_score_x = grad_score_x.unsqueeze(1)
    # grad_score_y = grad_score_y.unsqueeze(1)
    # Compute the similarity using the RBF kernel 
    kappa, grad_kappa = rbf_kernel(zeta, zeta)
    
    #kappa = kappa.squeeze(2)
    #grad_score = grad_score.unsqueeze(2)
    #a = grad_score * kappa
    #print(a)
    # c = a + b
    # print(c)
    # b = alpha * grad_kappa
    # print(b)
    #c = T.cat((a, b), dim=1)
    
    # grad_x = grad_score_x * kappa
    # grad_y = grad_score_y * kappa

    svgd = T.mean( T.matmul(grad_score.T, kappa.squeeze(-1)).T + alpha * grad_kappa, dim=1).detach()
    g_net.optimizer.zero_grad()
    autograd.backward(f_x, grad_tensors=svgd)
    g_net.optimizer.step()
  
#     z_i = g_net.forward((Variable(torch.FloatTensor(input)).view(-1,1).cpu()), num_particle=num_particle)
    
#     # N * k_upd * out_dim
#     z_j = g_net.forward((Variable(torch.FloatTensor(input)).view(-1,1).cpu()), num_particle=num_particle)
#     # N * k_fix * out_dim
    
    # # N * k_fix * out_dim
    # score = d_net(Variable(T.FloatTensor(input)).view(-1,1).cpu(), z_j)
    
    # grad_score = autograd.grad(score.sum(), z_j)[0].unsqueeze(2)
    
    # # N * k_fix * 1 * out_dim
    # kappa, grad_kappa = rbf_kernel(z_j, z_i)
    # # N * k_fix * k_upd / N * k_fix * k_upd * 1
    
    # svgd = T.mean( kappa*grad_score + alpha*grad_kappa, dim=1 ).detach()
    # # N * k_fix * k_upd * out_dim -> N * k_upd * out_dim
    
    # g_optim.zero_grad()
    # autograd.backward(
    #     -z_i,
    # grad_tensors=svgd)
    # g_optim.step()
    
def learn_D(g_net,d_net, x_obs, batch_size = 10, epsilon = 0.001):
    # Draw zeta random samples
    zeta = T.FloatTensor(batch_size, 2).uniform_(0, 1)
    # Forward the noise through the network
    f_x = g_net.forward(zeta)
    # Get the energy using the discriminator
    score = d_net.forward(f_x, x_obs)
    
    
    
    
    # # N * k_upd * 1
    # eps = T.rand(g_output.size()).cpu()
    # z_flow = (1-eps)*g_output + eps* Variable(T.FloatTensor(target)).view(-1,1,1).cpu()
    # # N * k_upd * 1

    # # energy calc
    # g_energy = d_net(Variable(T.FloatTensor(input)).view(-1,1).cpu(), g_output)
    # # N * k_upd * 1
    # data_energy = d_net(Variable(T.FloatTensor(input)).view(-1,1).cpu(),                        Variable(T.FloatTensor(target)).view(-1,1,1).cpu())
    # # N * 1 * 1
    # flow_energy = d_net(Variable(T.FloatTensor(input)).view(-1,1).cpu(), z_flow)
    # # N * k_upd * 1

    # loss = -data_energy.mean()+g_energy.mean()
    # grad = autograd.grad( flow_energy.sum(), z_flow ,retain_graph=True )[0].norm()
    # loss = loss + 10. * (grad-1)**2

    # d_optim.zero_grad()
    # autograd.backward(
    #     loss
    # )
    # d_optim.step()


# # Test it!

# In[ ]:


TRAIN_PARTICLES = 10
NUM_PARTICLES = 100
ITER_NUM = int(3e4)
BATCH_SIZE = 5
IMAGE_SHOW = 1e+3

def train(alpha=1.0):
    for i in range(ITER_NUM):
        # sample minibatch
        index = np.random.choice(range(len(x)), size=BATCH_SIZE, replace=False)
        mini_x = x[index]
        mini_y = y[index]
        
        x_obs = []
        for i in range(len(mini_x)):
            x_obs.append([mini_x[i], mini_y[i]])
    
        x_obs = T.from_numpy(np.array(x_obs)).float()
        
        # eval_svgd
        learn_G(g_net, d_net, x_obs, batch_size=BATCH_SIZE)
        
        # sample minibatch
        index = np.random.choice(range(len(x)), size=BATCH_SIZE, replace=False)
        mini_x = x[index]
        mini_y = y[index]
        
        x_obs = []
        for i in range(len(mini_x)):
            x_obs.append([mini_x[i], mini_y[i]])
    
        x_obs = T.from_numpy(np.array(x_obs)).float()
        
        # learn discriminator
        learn_D(g_net, d_net, x_obs, batch_size=BATCH_SIZE)

        if (i+1)%IMAGE_SHOW == 0:
            plt.rcParams["figure.figsize"] = (20,10)
            fig, ax = plt.subplots()
            plt.xlim(-10,10)
            plt.ylim(-30,30)
            x_test = np.linspace(-10,10, 300).reshape(-1,1)
                        
            ax.scatter(x, y,s=3,color='blue')
            
            predict = g_net.forward(Variable(T.FloatTensor(x_test)).cpu(),                                     num_particle=NUM_PARTICLES).detach().cpu().numpy()
            
            ax.scatter(list(np.linspace(-10,10, 300))*NUM_PARTICLES,                       predict.transpose(1,0,2).reshape(-1)                       ,s=1, alpha=0.4, color='red')
            
            ax.set_title('Iter:'+str(i+1)+' alpha:'+str(alpha))
            plt.show()

g_net = G().cpu()
d_net = D().cpu()
g_optim = T.optim.Adam(g_net.parameters(), lr=1e-3)
d_optim = T.optim.Adam(d_net.parameters(), lr=1e-4)
train(1.)


# In[ ]:




