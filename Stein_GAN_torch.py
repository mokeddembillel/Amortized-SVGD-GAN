import numpy as np
import seaborn as sns
sns.set()

import torch as T
import torch.autograd as autograd





print(T.__version__)
device = T.device('cuda' if T.cuda.is_available() else 'cpu')



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


def learn_G(P, g_net, d_net, batch_size = 10):
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
    
    
    
    
def learn_D(g_net,d_net, x_obs, batch_size = 10):
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
    
    loss_gp = T.mean(grad_r.norm(dim=1,p=2)**2) + T.mean(grad_f.norm(dim=1,p=2)**2)
    
    loss = data_score.mean() - gen_score.mean() + 10 * loss_gp
    
    #loss = data_score.mean() - gen_score.mean()
    #print("Loss : ", loss.detach().numpy())
    
    # Update the network
    d_net.optimizer.zero_grad()
    autograd.backward(loss)
    d_net.optimizer.step()
    
    return data_score.detach().numpy(), gen_score.detach().numpy(), loss.detach().numpy()
    