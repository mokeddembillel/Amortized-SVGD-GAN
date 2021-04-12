# In[ ]:


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


# plt.figure(figsize=(10,8))
# plt.xlim(-10,10)
# plt.ylim(-30,30)
# plt.scatter(x, y,s=3)


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
    loc = T.Tensor([[-5.0, 0.0], [5.0, 0.0]]).to(device)
    cov = T.Tensor([0.5, 0.5]).diag().unsqueeze(0).repeat(2, 1, 1).to(device)
  
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
    def __init__(self,lr=1e-3, input_dim=2):
        super().__init__()
        
        # Initialize Input dimentions
        self.input_dim = 2
        self.input_fc1_dim = 8
        self.fc1_output_dim = 8
        self.output_dim = 2
        
        # Define the NN layers
        self.input_layer = nn.Linear(self.input_dim, self.input_fc1_dim)
        self.fc1 = nn.BatchNorm1d(self.input_fc1_dim, self.fc1_output_dim)
        self.output_layer = nn.Linear(self.fc1_output_dim, self.output_dim)
        
        # Initialize layers weights
        self.input_layer.weight.data.normal_(0,1)
        #self.fc1.weight.data.normal_(0,0.2)
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
        self.input_dim = 2
        self.input_fc1_dim = 64
        self.fc1_output_dim = 64
        self.output_dim = 1
        
        # Define the NN layers
        self.input_layer = nn.Linear(self.input_dim, self.input_fc1_dim)
        #self.fc1 = nn.Linear(self.input_fc1_dim, self.fc1_output_dim)
        self.output_layer = nn.Linear(self.fc1_output_dim, self.output_dim)
        
        # Initialize layers weights
        self.input_layer.weight.data.normal_(0,1)
        #self.fc1.weight.data.normal_(0,0.2)
        self.output_layer.weight.data.normal_(0,1)
        
        # Initialize layers biases
        nn.init.constant_(self.input_layer.bias, 0.0)
        #nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)
        
        self.lr = lr
        # Define Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr = self.lr)
        
        # Set Device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, input):
        X = self.input_layer(input)
        X = F.relu(X)
        #X = self.fc1(X)
        #X = F.relu(X)
        output = self.output_layer(X)
        return F.sigmoid(output)

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


# def rbf_kernel(input_1, input_2,  h_min=1e-3):
    
#     k_fix, out_dim1 = input_1.size()[-2:]
    
#     k_upd, out_dim2 = input_2.size()[-2:]
    
#     assert out_dim1 == out_dim2

#     leading_shape = input_1.size()[:-2]
#     # Compute the pairwise distances of left and right particles.
#     diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)  
#     # N * k_fix * 1 * out_dim / N * 1 * k_upd * out_dim/ N * k_fix * k_upd * out_dim
#     dist_sq = diff.pow(2).sum(-1)
#     # N * k_fix * k_upd
#     dist_sq = dist_sq.unsqueeze(-1)
#     # N * k_fix * k_upd * 1
    
#     # Get median.
#     median_sq = T.median(dist_sq, dim=1)[0]
#     median_sq = median_sq.unsqueeze(1)
#     # N * 1 * k_upd * 1
    
#     h = median_sq / np.log(k_fix + 1.)
#     # N * 1 * k_upd * 1

#     kappa = T.exp(- dist_sq / h)
#     # N * k_fix * k_upd * 1

#     # Construct the gradient
#     kappa_grad = -2. * diff / h * kappa
#     return kappa, kappa_grad

def rbf_kernel(input_1, input_2,  h_min=1e-3):
    
    k_fix, out_dim1 = input_1.size()[-2:]
    k_upd, out_dim2 = input_2.size()[-2:]
    
    assert out_dim1 == out_dim2
    
    # Compute the pairwise distances of left and right particles.
    diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)  
    #print(diff)

    dist_sq = diff.pow(2).sum(-1)
    #print(dist_sq)
    dist_sq = dist_sq.unsqueeze(-1)
    #print(dist_sq)
    
    # Get median.
    median_sq = T.median(dist_sq, dim=1)[0]
    #print(median_sq)
    median_sq = median_sq.unsqueeze(1)
    #print(median_sq)
    
    h = median_sq / np.log(k_fix + 1.)
    #print(h)

    kappa = T.exp(- dist_sq / (h + 1e-8))
    #print(kappa)

    # Construct the gradient
    kappa_grad = -autograd.grad(kappa.sum(), input_1)[0]
    #print(kappa_grad)
    return kappa, kappa_grad


def learn_G(g_net, d_net, x_obs, batch_size = 10, alpha=1.):
    # Draw zeta random samples
    zeta = T.FloatTensor(batch_size, 2).uniform_(0, 1)
    zeta_u = T.FloatTensor(batch_size, 2).uniform_(0, 1)
    #zeta.requires_grad_(True)
    # Forward the noise through the network
    f_x = g_net.forward(zeta)
    f_x_u = g_net.forward(zeta_u)
    #print(f_x)
    ### Equation 7 (Compute Delta xi)
    # Get the energy using the discriminator
    score = d_net.forward(f_x_u)
    #print('scoooooooore :', score)
    #print(score)
    # Get the Gradients of the energy with respect to x and y
    grad_score = autograd.grad(-score.sum(), f_x_u)[0].squeeze(-1)
    #print(grad_score)
    # Compute the similarity using the RBF kernel 
    #print(zeta)
    kappa, grad_kappa = rbf_kernel(f_x_u, f_x) # <<<<<<<<<<<<<<<<<<<<<<<<
    # print(kappa)
    # print(grad_kappa)
    # grad_score_x = grad_score[:, 0].unsqueeze(-1)
    # grad_score_x = grad_score[:, 1].unsqueeze(-1)
    # print(kappa)
    # print(grad_kappa)

    
    
    svgd = (T.matmul(kappa.squeeze(-1), grad_score) - 1 * grad_kappa) / f_x.size(0)
    #svgd = T.mean(T.matmul(grad_score.T, kappa) - 1 * grad_kappa, dim=1).detach()

    # print(svgd)
   
    
    gen_grad_score = []
    for i in range(batch_size):
        grad = []
        for j in range(2):
            grad.append(list(autograd.grad(f_x[i, j], (g_net.parameters()), 
                                            retain_graph=True, allow_unused=True)))
            
            # print("hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
            #print(f_x[i, j])
            # print("****************************************")
            # print(grad[-1])
            if T.any(grad[-1][0].isnan()):
                raise Exception("NaNs Exception")
            #return
        gen_grad_score.append(grad)
    
    # print("ffffffffffffffffffffffffffffffffffffffffffffffffffffff")
    gradients = []
    for i in range(batch_size):
        grad = gen_grad_score[i]
        delta = svgd[i]
        # print(delta)
        #a = T.matmul()
        
        for j in range(len(grad[0])):
            # print('jjjjj')
            #print(grad[0][j])
            #print(delta[0])
            grad[0][j] = grad[0][j] * delta[0]
            #print(grad[0][j] * delta[0])

        for j in range(len(grad[1])):
            # print('rrrrr')
            # print(grad[1][j])
            # print(delta[1])
            # print(grad[1][j])
            grad[1][j] = grad[1][j] * delta[1]
            #print(grad[1][j] * delta[1])

        
        grads = []
        for j in range(len(grad[0])):
            # print('kkkkk')
            # print(grad[0][j])
            # print(grad[1][j])
            a = T.cat((grad[0][j].unsqueeze(0), grad[1][j].unsqueeze(0)),dim=0)
            # print(a)
            b = T.sum(a, dim=0)
            # print(b)
            grads.append(b.unsqueeze(0))
            
            # print(grads[-1])
        gradients.append(grads)
    
    # print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
    gradients_sum = []
    for i in range(len(gradients[0])):
        tmp = []
        for j in range(batch_size):
            tmp.append(gradients[j][i])
            
        a = T.cat(tuple(tmp), dim=0)
        b = T.sum(a, dim=0)
        gradients_sum.append(b)
    
    with T.no_grad():
        for i, p in enumerate(g_net.parameters()):
            #print(p)
            #print(gradients_sum[i])
            #print("gradients : ", gradients_sum[i])
            p.copy_(p + g_net.lr * gradients_sum[i])
            #print(p)
            if T.any(p.isnan()):
                raise Exception("NaNs Exception")
    
        
    
    # g_net.optimizer.zero_grad()
    # autograd.backward(-f_x, grad_tensors= svgd)
    # g_net.optimizer.step()
  
#     z_i = g_net.forward((Variable(T.FloatTensor(input)).view(-1,1).cpu()), num_particle=num_particle)
    
#     # N * k_upd * out_dim
#     z_j = g_net.forward((Variable(T.FloatTensor(input)).view(-1,1).cpu()), num_particle=num_particle)
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
    # Get the energy of the observed data using the discriminator
    data_score = d_net.forward(x_obs)
    # Get the energy of the generated data using the discriminator
    gen_score = d_net.forward(f_x)
    
    data_grad_score = []
    for i in range(batch_size):
        grad = list(autograd.grad(data_score[i], (d_net.parameters()), retain_graph=True, allow_unused=True, create_graph=True))
        for j in range(len(grad)):
            grad[j] = grad[j].unsqueeze(0)
        data_grad_score.append(grad)
    
    gen_grad_score = []
    for i in range(batch_size):
        grad = list(autograd.grad(gen_score[i], (d_net.parameters()), retain_graph=True, allow_unused=True, create_graph=True))
        for j in range(len(grad)):
            grad[j] = grad[j].unsqueeze(0)
        gen_grad_score.append(grad)
        
        
    data_grad_score_mean = []
    for i in range(len(data_grad_score[0])):
        tmp = []
        for t in data_grad_score:
            tmp.append(t[i])
        data_grad_score_mean.append(T.cat(tuple(tmp), dim=0).mean(0))
        
        
    gen_grad_score_mean = []
    for i in range(len(gen_grad_score[0])):
        tmp = []
        for t in gen_grad_score:
            tmp.append(t[i])
        gen_grad_score_mean.append(T.cat(tuple(tmp), dim=0).mean(0))
    
    gradients = -1 * data_grad_score_mean + gen_grad_score_mean
    
    with T.no_grad():
        for i, p in enumerate(d_net.parameters()):
            p.copy_(p + d_net.lr * gradients[i])
            # print(p)
     
    


# # Test it!

# In[ ]:


TRAIN_PARTICLES = 10
NUM_PARTICLES = 100
ITER_NUM = int(3e4)
BATCH_SIZE = 16
IMAGE_SHOW = 1e+3







  
mog2 = MoG2(device=device)

n = 300
X_init = (5 * T.randn(n, *mog2.event_shape)).to(device)

mog2_chart = get_density_chart(mog2, d=7.0, step=0.1)

X = X_init.clone()
svgd = SVGD(mog2, K, T.optim.Adam([X], lr=1e-1))
for _ in range(1000):
    svgd.step(X)

x = X[:, 0].reshape(-1, 1)
y = X[:, 1].reshape(-1, 1)



plt.rcParams["figure.figsize"] = (20,10)
fig, ax = plt.subplots()
plt.xlim(-30,30)
plt.ylim(-20,20)

zeta = T.FloatTensor(3000, 2).uniform_(-10, 10)

ax.scatter(x, y,s=3,color='blue')

ax.scatter(zeta[:, 0].numpy(), zeta[:, 1].numpy(),s=1, color='red')

ax.set_title('Iter:'+str(0)+' alpha:')
plt.show()
# print(x)
# print(y)
def train(alpha=1.0):
    for i in range(ITER_NUM):
        # sample minibatch
        index = np.random.choice(range(len(x)), size=BATCH_SIZE, replace=False)
        mini_x = x[index]
        mini_y = y[index]
        
        x_obs = []
        for j in range(len(mini_x)):
            x_obs.append([mini_x[j], mini_y[j]])
    
        x_obs = T.from_numpy(np.array(x_obs)).float()
        
        # eval_svgd
        learn_G(g_net, d_net, x_obs, batch_size=BATCH_SIZE)
        
        # sample minibatch
        index = np.random.choice(range(len(x)), size=BATCH_SIZE, replace=False)
        mini_x = x[index]
        mini_y = y[index]
        
        x_obs = []
        for j in range(len(mini_x)):
            x_obs.append([mini_x[j], mini_y[j]])
    
        x_obs = T.from_numpy(np.array(x_obs)).float()
        
        # learn discriminator
        learn_D(g_net, d_net, x_obs, batch_size=BATCH_SIZE)
        #print(i)
        if (i+1)%IMAGE_SHOW == 0:
            plt.rcParams["figure.figsize"] = (20,10)
            fig, ax = plt.subplots()
            plt.xlim(-30,30)
            plt.ylim(-20,20)
            
            zeta = T.FloatTensor(3000, 2).uniform_(-10, 10)

            ax.scatter(x, y,s=3,color='blue')
            
            predict = g_net.forward(zeta.cpu()).detach().cpu().squeeze(-1)
            
            # print(zeta)                        
            print(predict)

            
            ax.scatter(predict[:, 0].numpy(), predict[:, 1].numpy(),s=1, color='red')
            # ax.scatter(list(np.linspace(-10,10, 300)), list(np.linspace(-10,10, 300)),s=1, alpha=0.4, color='red')
            # ax.scatter(predict[:, 0], predict[:, 1],s=1, alpha=0.4, color='red')

            
            
            ax.set_title('Iter:'+str(i+1)+' alpha:'+str(alpha))
            plt.show()

g_net = G().cpu()
d_net = D().cpu()
g_optim = T.optim.Adam(g_net.parameters(), lr=1e-3)
d_optim = T.optim.Adam(d_net.parameters(), lr=1e-4)
train(1.)


# In[ ]:




