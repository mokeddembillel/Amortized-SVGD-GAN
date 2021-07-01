import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import torch as T

from utilities import generate_dist_1, generate_dist_2, generate_dist_3
from networks import G, D
from Stein_GAN_torch import learn_G, learn_D

print(T.__version__)
device = T.device('cuda' if T.cuda.is_available() else 'cpu')

ITER_NUM = int(6e4)
BATCH_SIZE = 64
IMAGE_SHOW = 5e+2

#x, y, mog2 = generate_dist_1()
x, y, mog2 = generate_dist_2()
#x, y, mog2 = generate_dist_3()


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
        # sample minibatch
        index = np.random.choice(range(len(x)), size=BATCH_SIZE, replace=False)
        mini_x = x[index]
        mini_y = y[index]
        x_obs = []
        for j in range(len(mini_x)):
            x_obs.append([mini_x[j], mini_y[j]])
        x_obs = T.from_numpy(np.array(x_obs, dtype=np.float32)).float()
        if i%1 == 0:
            
            # learn discriminator
            a, b, c = learn_D(g_net, d_net, x_obs, batch_size=BATCH_SIZE)
            data_score.append(a[0])
            gen_score.append(b[0])
            loss.append(c)
            #print(a)
        
        if i%20 == 0:
            
            # eval_svgd
            learn_G(mog2, g_net, d_net, batch_size=BATCH_SIZE)
        
        
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
