
import torch as T

import altair as alt
import numpy as np

import seaborn as sns
import pandas as pd
sns.set()
import torch.optim as optim


device = T.device('cuda' if T.cuda.is_available() else 'cpu')


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
    
    
    
class RBF(T.nn.Module):
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
    score_func = T.autograd.grad(log_prob.sum(), X)[0]

    K_XX = self.K(X, X.detach())
    grad_K = -T.autograd.grad(K_XX.sum(), X)[0]

    phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

    return phi

  def step(self, X):
    self.optim.zero_grad()
    X.grad = -self.phi(X)
    self.optim.step()
  
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

def generate_dist_1():
    f = lambda x : x**2

    x = np.linspace(-3,3,300)
    #y = f(x) + np.random.uniform(low=-1.0,high=1.0,size=x.shape)* 4* np.cos(x/2+.5)
    y = f(x) + np.random.normal(0,1,size=x.shape) + np.random.normal(1,1,size=x.shape)
    
    y_ = -y
    x = np.concatenate((x,x), axis=0)
    y = np.concatenate((y,y_),axis =0)
    return x, y, None

def generate_dist_2():
    mog2 = MoG2(device=device)

    n = 300
    X_init = (10 * T.randn(n, *mog2.event_shape)).to(device)
    
    #mog2_chart = get_density_chart(mog2, d=7.0, step=0.1)
    
    X = X_init.clone()
    svgd = SVGD(mog2, K, T.optim.Adam([X], lr=1e-1))
    for _ in range(1500):
        svgd.step(X)
    
    x = X[:, 0].reshape(-1, 1)
    y = X[:, 1].reshape(-1, 1)
    return x, y, mog2

def generate_dist_3():
    gauss = T.distributions.MultivariateNormal(T.Tensor([10, 10]).to(device),
        covariance_matrix=5 * T.Tensor([[0.3, 0],[0, 0.3]]).to(device))
    n = 300
    X_init = (3 * T.randn(n, *gauss.event_shape)).to(device)
    X = X_init.clone()
    svgd = SVGD(gauss, K, optim.Adam([X], lr=1e-1))
    for _ in range(1000):
        svgd.step(X)
    x = X[:, 0].reshape(-1, 1)
    y = X[:, 1].reshape(-1, 1)
    return x, y, None
















