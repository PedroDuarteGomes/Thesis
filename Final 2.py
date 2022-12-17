#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
from scipy.stats import norm
from numpy import linalg as la
import numpy.random as npr
from tabulate import tabulate
from matplotlib import pyplot as plt
import random
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import optim
from scipy.stats import truncnorm


# In[59]:


def BlackScholes(S0,r,sigma,T,K):
    d1 =  1 / (sigma * np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    d2 = d1 - sigma * np.sqrt(T)
    return norm.cdf(d1) * S0 - norm.cdf(d2) * K * np.exp(-r*T)

def BlackScholesCallDelta(S0,r,sigma,T,K):
    d1 =  1 / (sigma * np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    return norm.cdf(d1)

BlackScholes_vec=np.vectorize(BlackScholes)

BlackScholesCallDelta_vec=np.vectorize(BlackScholesCallDelta)


# In[60]:


nSimul = 32768
T1 = 1.0
T2 = 2.0
K = 110.0

spot = 100.0
vol = 0.2
vol0 = 0.5 # vol is increased over the 1st period so we have more points in the wings


# simulate all Gaussian returns (N1, N2) first
# returns: matrix of shape [nSimul, TimeSteps=2]
returns = np.random.normal(size=[nSimul,2])

# generate paths, step by step, and not path by path as customary
# this is to avoid slow Python loops, using NumPy's optimized vector functions instead

# generate the vector of all scenarios for S1, of shape [nSimul]
S1 = spot * np.exp(-0.5*vol0*vol0*T1 + vol0*np.sqrt(T1)*returns[:,0])

# generate the vector of all scenarios for S2, of shape [nSimul]
S2 = S1 * np.exp(-0.5*vol*vol*(T2-T1) + vol*np.sqrt(T2-T1)*returns[:,1])

# training set, X and Y are both vectors of shape [nSimul]
X = S1
Y = np.maximum(0, S2 - K)
xAxis = np.linspace(20, 200, 100)

xAxis=xAxis.reshape(-1,1)


# In[61]:


#Normaize the data so, that the activation function can act on those values.
def normal(x):
    meanx=np.mean(x)
    stdx=np.std(x)
    
    norm=(x - meanx) / stdx
    
    return norm.reshape(-1,1),meanx,stdx


# In[62]:


class NeuralNetwork(nn.Module):
    def __init__(self,inputsize,outputsize):
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        torch.manual_seed(123)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputsize,3),
            nn.ELU(),
            nn.Linear(3, 5),
            nn.ELU(),
            nn.Linear(5,3), 
            nn.ELU(),
            nn.Linear(3,outputsize),
        )
        
        
        #w = torch.empty(0,1)
        #nn.init.normal_(w)
    
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# In[63]:


inputDim = 1       # takes variable 'x' 
outputDim = 1       # takes variable 'y'
learningRate = 0.05
epochs = 100
#weight=torch.empty(3)
torch.manual_seed(123)
model = NeuralNetwork(inputDim, outputDim)
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()
    


# In[66]:


#Adam optmization
criterion = torch.nn.MSELoss() 
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.05)


# In[67]:


def Train(x,y):
 
    
    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x).cuda().float())
            labels = Variable(torch.from_numpy(y).cuda().float())
        else:
            inputs = Variable(torch.from_numpy(x).float())
            labels = Variable(torch.from_numpy(y).float())

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

    # get output from the model, given the inputs
        outputs = model(inputs)

    # get loss for the predicted output
        loss = criterion(outputs, labels)
        #print(loss)
    # get gradients w.r.t to parameters
        loss.backward()

    # update parameters
        optimizer.step()

        #print('epoch {}, loss {}'.format(epoch, loss.item()))


# In[68]:


def predict(xs,S,P):
    # first, normalize
    nxs = (xs - (normal(x=S)[1])) / (normal(x=S)[2])
    # forward feed through ANN
    # we don't need gradients in the testing phase
    with torch.no_grad():
        if torch.cuda.is_available():
            nys = model(Variable(torch.from_numpy(nxs.reshape(-1,1)).cuda().float())).cpu().data.numpy()
        else:
            nys = model(Variable(torch.from_numpy(nxs.reshape(-1,1))).float()).data.numpy()
    
    # de-normalize output
    ys = (normal(x=P)[1]) + (normal(x=P)[2]) * nys
    # we get a matrix of shape [size of xs][1], which we reshape as vector [size of xs]
    return np.reshape(ys, [-1])


# In[69]:



Train(normal(X)[0],normal(Y)[0])


# In[70]:





BS_price=BS_prices=BlackScholes_vec(S0=xAxis,r=0,sigma=0.2,T=1.0,K=110.0)
predicted=predict(xAxis,X,Y)

S1=1
#line_learn = plt.plot(Sval,y,label="Deep Neural Net")
line_learn = plt.plot(xAxis,predicted,label="Neural Regression")
line_BS = plt.plot(xAxis,BS_price, label="Black-Scholes")

plt.xlabel("Spot Price")
plt.ylabel("Option Price")
#plt.title(r'Time: %1.1f' % time, loc='left', fontsize=11)
#plt.title(r'Strike: %1.2f' % K, loc='right', fontsize=11)
#plt.title(r'Initial price: %1.2f' % S1, loc='center', fontsize=11)
plt.legend()
plt.savefig("neural.png")
plt.show()
#plt.savefig("deephedge.png", dpi=150)


# In[71]:


Prices_rg_mc_diff=[]

for i in range(len(xAxis)-1):
    delta=(predicted[i+1]-predicted[i])/(xAxis[i+1]-xAxis[i])
    Prices_rg_mc_diff.append(delta) 


# In[72]:


BS_delta=BlackScholesCallDelta(S0=xAxis,r=0,sigma=0.2,T=1.0,K=110.0)
predicted=predict(xAxis,X,Y)

S1=1
#line_learn = plt.plot(Sval,y,label="Deep Neural Net")
line_learn = plt.plot(xAxis[1:],Prices_rg_mc_diff,label="Neural Regression")
line_BS = plt.plot(xAxis[1:],BS_delta[1:], label="Black-Scholes")

plt.xlabel("Spot Price")
plt.ylabel("Option Price")
#plt.title(r'Time: %1.1f' % time, loc='left', fontsize=11)
plt.title(r'Strike: %1.2f' % K, loc='right', fontsize=11)
plt.title(r'Initial price: %1.2f' % S1, loc='center', fontsize=11)
plt.legend()
plt.show()
#plt.savefig("deephedge.png", dpi=150)
plt.savefig("deephedge.pdf")


# In[40]:


T=1
n=1000
K=100
#sigma=0.2
r=0
S0=100
#S_T=S0
steps=52
dt=T/steps
vol0=0.2
vol=0.2


# In[41]:


np.random.seed(123)
i=1
T_=T-(i-1)*dt
vS0=K*np.exp(-0.5*vol0*vol0*dt + vol0*np.sqrt(T)*npr.normal(0,1,(n,1))[:,0])
vS_T=vS0*np.exp(-0.5*vol*vol*dt + vol*np.sqrt(T-(i-1)*dt)*npr.normal(0,1,(n,1))[:,0])
vC=np.maximum(vS_T-K,0)


# In[42]:


torch.manual_seed(123)
Train(normal(vS0)[0],normal(vC)[0])


# In[43]:


#Compute price
V0=predict(S0,vS0,vC)#0.08914879(closer than other attempts :)
print(BlackScholes_vec(S0,r,vol,T,K))#0.07965567
print(V0)


# In[44]:


#Compute first delta
at=BlackScholesCallDelta_vec(S0,r,vol,T,K)
bt=V0-at*S0
at
bt
vS_T1=S0


# In[45]:


#Other FD method
torch.manual_seed(123)
vS_T1=S0
np.random.seed(123)
epsilon=0.999
for i in range(2,steps):
    
    #Training the Neural Network
    vS0=K*np.exp(-0.5*vol0*vol0*dt + vol0*np.sqrt(T)*npr.normal(0,1,(n,1))[:,0])
    vS_T=vS0*np.exp(-0.5*vol*vol*dt + vol*np.sqrt(T-(i-1)*dt)*npr.normal(0,1,(n,1))[:,0])
    vC=np.maximum(vS_T-K,0)
    Train(normal(vS0)[0],normal(vC)[0])
    #Evaluate:
    vS_T1=vS_T1*np.exp(-0.5*vol0*vol0*dt+np.sqrt(dt)*vol0*npr.normal(0,1,(n,1))[:,0])
    p=predict(vS_T1,vS0,vC)
    p_eps=predict(vS_T1+epsilon,vS0,vC)
    Vt=at*vS_T1+bt
    at=(p_eps-p)/epsilon
    bt=Vt-at*vS_T1
    
    
    print(i)
    print(np.max(at))
    print(np.max(Vt))


# In[46]:


vPnL=(Vt-np.maximum(vS_T1-K,0))/V0
#vPnL=(Vt-np.maximum(vS_T1-K,0))
print(np.std(vPnL))
plt.hist(vPnL,bins=75)
print(np.std(vPnL))
#np.mean(vPnL)



# In[47]:


T=1
n=1000
K=100
sigma=0.2
r=0
S0=100
S_T=S0
steps=52
dt=T/steps


# In[48]:


V_0=BlackScholes_vec(S0,r,sigma,T,K)
a_t=BlackScholesCallDelta(S0,r,sigma,T,K)
b_t=V_0-a_t*S0

np.random.seed(123)
#V_t=[]
for i in range(steps):
    S_T=S_T*np.exp(-0.5*sigma*sigma*dt+np.sqrt(dt)*sigma*npr.normal(0,1,(n,1)))
    #S_T=S_T*np.exp(0.1*dt+np.sqrt(dt)*sigma*npr.normal(0,1,(n,1)))
    V_t=a_t*S_T+b_t
    a_t=BlackScholesCallDelta(S_T,r,sigma,T-i*dt,K)
    b_t=V_t-a_t*S_T


# In[49]:


vPnL_bs=(V_t-np.maximum(S_T-K,0))/V_0


# In[50]:


vPnL_bs=vPnL_bs.reshape(1,-1)
plt.style.use('seaborn-deep')
plt.hist(vPnL_bs[0],bins=75)
np.std(vPnL_bs)


# In[53]:


plt.hist([vPnL_bs[0], vPnL], bins=75, label=['Black-Scholes', 'LSMC Neural '])
plt.legend(loc='upper right')
plt.savefig("Neural reg vs Black-Scholes")
plt.show()

