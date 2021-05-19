
# In[ ]:

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()

import torch as T
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F


import math
import numpy as np
import torch.optim as optim
import altair as alt

# In[ ]:


import torch 
print(torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class MoG(T.distributions.Distribution):
  def __init__(self, loc, covariance_matrix):
    self.num_components = loc.size(0)
    self.loc = loc
    self.covariance_matrix = covariance_matrix

    self.dists = [
      T.distributions.MultivariateNormal(mu, covariance_matrix=sigma)
      for mu, sigma in zip(loc, covariance_matrix)
    ]
    
    super(MoG, self).__init__(T.Size([]), T.Size([loc.size(-1)]))

  @property
  def arg_constraints(self):
    return self.dists[0].arg_constraints

  @property
  def support(self):
    return self.dists[0].support

  @property
  def has_rsample(self):
    return False

  def log_prob(self, value):
    return T.cat(
      [p.log_prob(value).unsqueeze(-1) for p in self.dists], dim=-1).logsumexp(dim=-1)

  def enumerate_support(self):
    return self.dists[0].enumerate_support()

class MoG2(MoG):
  def __init__(self, device=None):
    loc = T.Tensor([[-10.0, 0.0], [10.0, 0.0]]).to(device)
    cov = T.Tensor([0.5, 0.5]).diag().unsqueeze(0).repeat(2, 1, 1).to(device)
    #print(cov.T)
    super(MoG2, self).__init__(loc, cov)
  
def get_density_chart(P, d=7.0, step=0.1):
  xv, yv = T.meshgrid([
      T.arange(-d, d, step), 
      T.arange(-d, d, step)
  ])
  pos_xy = T.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1)
  p_xy = P.log_prob(pos_xy.to(device)).exp().unsqueeze(-1).cpu()
  
  df = T.cat([pos_xy, p_xy], dim=-1).numpy()
  df = pd.DataFrame({
      'x': df[:, :, 0].ravel(),
      'y': df[:, :, 1].ravel(),
      'p': df[:, :, 2].ravel(),
  })
  
  chart = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q',
    color=alt.Color('p:Q', scale=alt.Scale(scheme='viridis')),
    tooltip=['x','y','p']
  )
  
  return chart


def get_particles_chart(X):
  df = pd.DataFrame({
      'x': X[:, 0],
      'y': X[:, 1],
  })

  chart = alt.Chart(df).mark_circle(color='red').encode(
    x='x:Q',
    y='y:Q'
  )
  
  return chart
# # Define net

# In[ ]:


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
        nn.init.constant_(self.fc1.bias, 1)
        nn.init.constant_(self.fc2.bias, 1)
        nn.init.constant_(self.fc3.bias, 1)
        nn.init.constant_(self.fc4.bias, 1)
        nn.init.constant_(self.fc5.bias, 1)
        
        
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
        
# class D(nn.Module):
#     def __init__(self,lr=1e-4, input_dim=2):
#         super().__init__()
        
#         # Initialize Input dimentions
#         self.fc1_dim = 2
#         self.fc2_dim = 128
#         self.fc3_dim = 1
        
#         #self.fc1 = nn.Linear(self.fc1_dim, self.fc3_dim)
        
#         # Define the NN layers
#         self.fc1 = nn.Linear(self.fc1_dim, self.fc2_dim)
#         self.fc2 = nn.Linear(self.fc2_dim, self.fc2_dim)
#         self.fc3 = nn.Linear(self.fc2_dim, self.fc2_dim)
#         self.fc4 = nn.Linear(self.fc2_dim, self.fc2_dim)
#         self.fc5 = nn.Linear(self.fc2_dim, self.fc3_dim)
        
#         # self.bn1 = nn.BatchNorm1d(self.fc2_dim)
#         # self.bn2 = nn.BatchNorm1d(self.fc2_dim)
        
#         # Initialize layers weights
#         self.fc1.weight.data.normal_(0,0.2)
#         self.fc2.weight.data.normal_(0,0.2)
#         self.fc3.weight.data.normal_(0,0.2)
#         self.fc4.weight.data.normal_(0,0.2)
#         self.fc5.weight.data.normal_(0,0.2)
        
#         # # Initialize layers biases
#         nn.init.constant_(self.fc1.bias, 0.0)
#         nn.init.constant_(self.fc2.bias, 0.0)
#         nn.init.constant_(self.fc3.bias, 0.0)
#         nn.init.constant_(self.fc4.bias, 0.0)
#         nn.init.constant_(self.fc5.bias, 0.0)
    
#         self.lr = lr
#         # Define Optimizer
#         self.optimizer = T.optim.Adam(self.parameters(), lr = self.lr, betas=(0.0, 0.99))
        
        
#         # Set Device
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)
        
#     def forward(self, input):
        
#         X = F.relu(self.fc1(input))
#         X = F.relu(self.fc2(X))
#         X = F.relu(self.fc3(X))
#         X = F.relu(self.fc4(X))
#         X = self.fc5(X)
#         return -T.log(F.sigmoid(X))

class RBF(torch.nn.Module):
  def __init__(self, sigma=None):
    super(RBF, self).__init__()

    self.sigma = sigma

  def forward(self, X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

    # Apply the median heuristic (PyTorch does not give true median)
    if self.sigma is None:
      np_dnorm2 = dnorm2.detach().cpu().numpy()
      h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
      sigma = np.sqrt(h).item()
    else:
      sigma = self.sigma

    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()

    return K_XY
  
# Let us initialize a reusable instance right away.
K = RBF()

class SVGD:
  def __init__(self, P, K, optimizer):
    self.P = P
    self.K = K
    self.optim = optimizer

  def phi(self, X):
    X = X.detach().requires_grad_(True)

    log_prob = self.P.log_prob(X)
    score_func = autograd.grad(log_prob.sum(), X)[0]

    K_XX = self.K(X, X.detach())
    grad_K = -autograd.grad(K_XX.sum(), X)[0]

    phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

    return phi

  def step(self, X):
    self.optim.zero_grad()
    X.grad = -self.phi(X)
    self.optim.step()


# # Calculate the amortized Stein Variational Gradient
# 
# $$
# \begin{eqnarray}
# \eta^{t+1} &\leftarrow \eta^{t} + \epsilon \frac{1}{m}\sum_{i=1}^m \partial_{\eta}f(\xi_i ; \eta^t) \phi^* (z_i)\\
# &\leftarrow \eta^{t} + \epsilon \frac{1}{nm} \sum_{i=1}^m \partial_{\eta}f(\xi_i ; \eta^t) \sum_{j=1}^{n} \big[ \nabla_{z_j} \log p(z_j) k(z_j, z_i) + \nabla_{z_j}k(z_j,z_i) \big]\\
# \end{eqnarray}
# $$

# In[ ]:


def rbf_kernel(X, Y,  h_min=1e-3):

    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    dnorm2 = XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0) - 2 * XY

    # Apply the median heuristic (PyTorch does not give true median)
    np_dnorm2 = dnorm2.detach().cpu().numpy()
    h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
    sigma = np.sqrt(h).item()
    
    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()
    
    grad_K = -autograd.grad(K_XY.mean(), X)[0]
    
    return K_XY, grad_K



def learn_G(P, g_net, d_net, x_obs, batch_size = 10, alpha=1.):
    # Draw zeta random samples
    zeta = T.FloatTensor(batch_size, 2).uniform_(0, 1)
    # Forward the noise through the network
    f_x = g_net.forward(zeta)
    ### Equation 7 (Compute Delta xi)
    # Get the energy using the discriminator
    score = d_net.forward(f_x)
    
    # Get the Gradients of the energy with respect to x and y
    grad_score = autograd.grad(-score.sum(), f_x)[0].squeeze(-1)
    #grad_score = autograd.grad(P.log_prob(f_x).sum(), f_x)[0].squeeze(-1)

    # Compute the similarity using the RBF kernel 
    kappa, grad_kappa = rbf_kernel(f_x, f_x)
    
    # Compute the SVGD
    svgd = (T.matmul(kappa.squeeze(-1), grad_score) + grad_kappa) / f_x.size(0)
    
    # Update the network
    g_net.optimizer.zero_grad()
    autograd.backward(-f_x, grad_tensors=svgd)
    g_net.optimizer.step()  
    
    
    
    
def learn_D(g_net,d_net, x_obs, batch_size = 10, epsilon = 0.001):
    # Draw zeta random samples
    zeta = T.FloatTensor(batch_size, 2).uniform_(0, 1)
    # Forward the noise through the network
    x_obs.requires_grad_(True)
    f_x = g_net.forward(zeta).requires_grad_(True)
    # Get the energy of the observed data using the discriminator
    data_score = d_net.forward(x_obs)
    # Get the energy of the generated data using the discriminator
    gen_score = d_net.forward(f_x)
    print("Data Score : ", data_score.mean().detach().numpy(), "\nGen Score : ", gen_score.mean().detach().numpy())

    #Calculate the GP loss
    grad_r = autograd.grad(data_score.sum(), x_obs,
                        allow_unused=True, 
                        create_graph=True, 
                        retain_graph=True)[0]
    grad_f = autograd.grad(gen_score.sum(), f_x,
                        allow_unused=True, 
                        create_graph=True, 
                        retain_graph=True)[0]
    loss_gp = torch.mean(grad_r.norm(dim=1,p=2)**2) + torch.mean(grad_f.norm(dim=1,p=2)**2)
    
    loss = data_score.mean() - gen_score.mean() + 10 * loss_gp
    
    #loss = data_score.mean() - gen_score.mean()
    #print("Loss : ", loss.detach().numpy())
    
    # Update the network
    d_net.optimizer.zero_grad()
    autograd.backward(loss)
    d_net.optimizer.step()
    
    return data_score.detach().numpy(), gen_score.detach().numpy(), loss.detach().numpy()
    


# # Test it!

# In[ ]:


TRAIN_PARTICLES = 10
NUM_PARTICLES = 100
ITER_NUM = int(6e4)
BATCH_SIZE = 64
IMAGE_SHOW = 5e+2
  
# mog2 = MoG2(device=device)

# n = 300
# X_init = (10 * T.randn(n, *mog2.event_shape)).to(device)

# #mog2_chart = get_density_chart(mog2, d=7.0, step=0.1)

# X = X_init.clone()
# svgd = SVGD(mog2, K, T.optim.Adam([X], lr=1e-1))
# for _ in range(1500):
#     svgd.step(X)

# x = X[:, 0].reshape(-1, 1)
# y = X[:, 1].reshape(-1, 1)




# gauss = torch.distributions.MultivariateNormal(torch.Tensor([10, 10]).to(device),
#         covariance_matrix=5 * torch.Tensor([[0.3, 0],[0, 0.3]]).to(device))
# n = 300
# X_init = (3 * torch.randn(n, *gauss.event_shape)).to(device)
# X = X_init.clone()
# svgd = SVGD(gauss, K, optim.Adam([X], lr=1e-1))
# for _ in range(1000):
#     svgd.step(X)
# x = X[:, 0].reshape(-1, 1)
# y = X[:, 1].reshape(-1, 1)
mog2 = None




plt.rcParams["figure.figsize"] = (20,10)
fig, ax = plt.subplots()
plt.xlim(-30,30)
plt.ylim(-20,20)

zeta = T.FloatTensor(3000, 2).normal_(0, 3)

ax.scatter(x, y,s=1,color='blue')

ax.scatter(zeta[:, 0].numpy(), zeta[:, 1].numpy(),s=1, color='red')

ax.set_title('Iter:'+str(0)+' alpha:')
plt.show()

def train(alpha=1.0):
        
    data_score, gen_score, loss = [], [], []
    for i in range(ITER_NUM):
        #('Iteration :', i)
        if i%1 == 0:
            # sample minibatch
            index = np.random.choice(range(len(x)), size=BATCH_SIZE, replace=False)
            mini_x = x[index]
            mini_y = y[index]
            x_obs = []
            for j in range(len(mini_x)):
                x_obs.append([mini_x[j], mini_y[j]])
            x_obs = T.from_numpy(np.array(x_obs)).float()
            # learn discriminator
            a, b, c = learn_D(g_net, d_net, x_obs, batch_size=BATCH_SIZE)
            data_score.append(a[0])
            gen_score.append(b[0])
            loss.append(c)
            #print(a)
        
        if i%20 == 0:
            # sample minibatch
            index = np.random.choice(range(len(x)), size=BATCH_SIZE, replace=False)
            mini_x = x[index]
            mini_y = y[index]
            x_obs = []
            for j in range(len(mini_x)):
                x_obs.append([mini_x[j], mini_y[j]])
            x_obs = T.from_numpy(np.array(x_obs)).float()
            # eval_svgd
            learn_G(mog2, g_net, d_net, x_obs, batch_size=BATCH_SIZE)
        
        
        #print(i)
        if (i+1)%IMAGE_SHOW == 0:
            plt.rcParams["figure.figsize"] = (20,10)
            fig, ax = plt.subplots()
            plt.xlim(-30,30)
            plt.ylim(-20,20)
            
            zeta = T.FloatTensor(1000, 2).normal_(0, 20)

            ax.scatter(x, y,s=1,color='blue')
            
            predict = g_net.forward(zeta.cpu()).detach().cpu().squeeze(-1)
            
            
            ax.scatter(predict[:, 0].numpy(), predict[:, 1].numpy(),s=1, color='red')
            
            
            ax.set_title('Iter:'+str(i+1)+' alpha:'+str(alpha))
            plt.show()
             
    def moving_average(a, n=10) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    data_score = moving_average(data_score, 5)
    gen_score = moving_average(gen_score, 5)
    loss = moving_average(loss, 5)
    
    
    
    plt.plot(data_score)
    plt.ylabel('Data score')
    plt.show()
    
    plt.plot(gen_score)
    plt.ylabel('Gen score')
    plt.show()
    
    plt.plot(loss)
    plt.ylabel('Loss')
    plt.show()

g_net = G().cpu()
d_net = D().cpu()

train(1.)

x = np.arange(-20, 20, 0.5)
y = np.zeros(len(x))
particles = []
for i in range(len(x)):
    particles.append([x[i], y[i]])
particles = T.tensor(np.array(particles, dtype=np.float32))

energy = d_net.forward(particles).detach().numpy()
#energy = mog2.log_prob(particles).detach().numpy()
#P.log_prob(f_x)

#print(energy)

plt.plot(energy)
plt.ylabel('energy')
plt.show()


# In[ ]:




