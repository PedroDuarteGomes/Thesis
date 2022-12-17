#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy.stats import norm
from numpy import linalg as la
import numpy.random as npr
from tabulate import tabulate
from matplotlib import pyplot as plt
import random


# In[3]:


def BlackScholes(S0,r,sigma,T,K):
    d1 =  1 / (sigma * np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    d2 = d1 - sigma * np.sqrt(T)
    return norm.cdf(d1) * S0 - norm.cdf(d2) * K * np.exp(-r*T)

def BlackScholesCallDelta(S0,r,sigma,T,K):
    d1 =  1 / (sigma * np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    return norm.cdf(d1)

BlackScholes_vec=np.vectorize(BlackScholes)

BlackScholesCallDelta_vec=np.vectorize(BlackScholesCallDelta)


# In[6]:


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


# In[7]:


def training(X,Y,nSimul):
    #Compute mx
    w=0.5
    p=7
    x0 = np.ones(nSimul)
    S=X
    S_2=X**2
    S_3=X**3
    S_4=X**4
    S_5=X**5
    S_6=X**6
    S_7=X**7
    mX=np.column_stack((x0,S, S_2, S_3,S_4,S_5,S_6,S_7))



    #Compute my
    x_z=np.zeros(len(X))
    x0=np.ones(len(X)) 
    S_1_val=2*X
    S_2_val=3*X**2
    S_3_val=4*X**3
    S_4_val=5*X**4
    S_5_val=6*X**5
    S_6_val=7*X**6
    #p=np.array(range(1,8))
    mY=np.column_stack((x_z,x0,S_1_val,S_2_val,S_3_val,S_4_val,S_5_val,S_6_val))
    vDelta=Y>K
    vtheta_deltaforce=la.inv((w*mX.T@mX + (1-w)*mY.T@mY))@(w*mX.T@Y+(1-w)*mY.T@vDelta)
    
    return vtheta_deltaforce


# In[65]:




def eveluate(xAxis,vtheta_deltaforce):
    x0=np.ones(len(xAxis)) 
    S_1_val=xAxis
    S_2_val=xAxis**2
    S_3_val=xAxis**3
    S_4_val=xAxis**4
    S_5_val=xAxis**5
    S_6_val=xAxis**6
    S_7_val=xAxis**7
    p=np.array(range(1,8))
    S_val_stack=np.column_stack((x0,S_1_val,S_2_val,S_3_val,S_4_val,S_5_val,S_6_val))
    ols_delta=vtheta_deltaforce[1:]
    olsdelta=np.array([ols_delta[i]*p[i] for i in range(len(p))])
    y= S_val_stack@olsdelta
    
    return y


# In[8]:


vtheta_deltaforce= list(reversed(training(X,Y,nSimul)))


# In[9]:


y=np.polyval(vtheta_deltaforce,xAxis)


# In[10]:


BS_prices=BlackScholes_vec(S0=xAxis,r=0,sigma=0.2,T=1.0,K=110.0)

S1=1
#line_learn = plt.plot(Sval,y,label="Deep Neural Net")
line_learn = plt.plot(xAxis,y,label="Differetial Polynomial")
line_BS = plt.plot(xAxis,BS_prices, label="Bachelier")
plt.xlabel("Spot Price")
plt.ylabel("Option Price")
#plt.title(r'Time: %1.1f' % time, loc='left', fontsize=11)
plt.legend()
plt.savefig("Differential Polynomial vs Black-Scholes")
plt.show()
#plt.savefig("deephedge.png", dpi=150)
#plt.savefig("deephedge.pdf")


# In[13]:


vtheta_deltaforce=training(X,Y,nSimul)

x0=np.ones(len(xAxis)) 
S_1_val=xAxis
S_2_val=xAxis**2
S_3_val=xAxis**3
S_4_val=xAxis**4
S_5_val=xAxis**5
S_6_val=xAxis**6
S_7_val=xAxis**7
p=np.array(range(1,8))
S_val_stack=np.column_stack((x0,S_1_val,S_2_val,S_3_val,S_4_val,S_5_val,S_6_val))
ols_delta=vtheta_deltaforce[1:]
olsdelta=np.array([ols_delta[i]*p[i] for i in range(len(p))])
y= S_val_stack@olsdelta


# In[14]:


BS_delta=BlackScholesCallDelta(S0=xAxis,r=0,sigma=0.2,T=1.0,K=110.0)
S1=1
#line_learn = plt.plot(xAxis,Prices_rg_mc_diff,label="Deep Neural Net")
line_learn = plt.plot(xAxis,y,label="Differetial Polynomial")
line_BS = plt.plot(xAxis,BS_delta, label="Black-Scholes")

plt.xlabel("Spot Price")
plt.ylabel("Delta")
#plt.title(r'Time: %1.1f' % time, loc='left', fontsize=11)
#plt.title(r'Strike: %1.2f' % K, loc='right', fontsize=11)
#plt.title(r'Initial price: %1.2f' % S1, loc='center', fontsize=11)
plt.legend()
plt.savefig("delta-ridge", dpi=150)
plt.show()


# In[3]:


T=1
n=1000
K=100
sigma=0.2
r=0
S0=100
S_T=S0
steps=52
dt=T/steps


# In[4]:


#Important.............
np.random.seed(123)
vol0=0.2
vol=0.2


ols_steps=[]
ols_delta=[]
for i in range(1,steps):
    T_=T-(i-1)*dt
    #vS0=K+np.sqrt(T)*sigma*npr.normal(0,1,(n,1))
    #vS_T= vS0+np.sqrt(T_)*sigma*npr.normal(0,1,(n,1))
    #vS0=K*np.exp(-0.5*vol0*vol0*dt + vol0*np.sqrt(T)*npr.normal(0,1,(n,1)))
    #vS_T=vS0*np.exp(-0.5*vol*vol*dt + vol*np.sqrt(T_)*npr.normal(0,1,(n,1)))
    
    vS0=S0*np.exp(-0.5*vol0*vol0*dt + vol0*np.sqrt(T)*npr.normal(0,1,(n,1)))
    vS_T=vS0*np.exp(-0.5*vol*vol*dt + vol*np.sqrt(T_)*npr.normal(0,1,(n,1)))
    
    #returns = np.random.normal(size=[n,2])
    #vS0 = S0 * np.exp(-0.5*vol0*vol0*T + vol0*np.sqrt(T)*returns[:,0])
    #vS_T=vS0*np.exp(-0.5*vol*vol*dt + vol*np.sqrt(T_)*returns[:,0])
    vC=np.maximum(vS_T-K,0)
    #regression 
    ols=training(vS0[:,0],vC[:,0],n)
    #ols=np.polyfit(vS0,vC,7)
    #Store the pricing coefficiens for each period
    ols_steps.append(ols)
    #regression delta
    p=np.array([7,6,5,4,3,2,1])
    olsdelta=ols[0:7]*p
    #Store the deltas for each period
    ols_delta.append(olsdelta)


# In[5]:


len(olsdelta)


# In[7]:


V_0=np.polyval(ols_steps[0],S0)
a_t=np.polyval(ols_delta[0],S0)
b_t=V_0-a_t*S0
vS_T=S0


dt=T/steps
np.random.seed(123)
for i in range(1,steps-2):
    vS_T=S_T*np.exp(-0.5*vol0*vol0*dt+np.sqrt(dt)*vol0*npr.normal(0,1,(n,1)))
    #vS_T=vS0*np.exp(-0.5*vol*vol*dt + vol*np.sqrt(dt)*returns[:,0])
    V_t=a_t*vS_T+b_t
    a_t=np.polyval(list(reversed(ols_delta[i+1])),vS_T)
    #a_t=eveluate(vS_T[:,0],ols_steps[i+1])
    b_t=V_t-a_t*vS_T
    


# In[8]:


vPnL=(V_t-np.maximum(vS_T-K,0))/V_0


# In[9]:


vPnL=vPnL.reshape(1,-1)
plt.style.use('seaborn-deep')
plt.hist(vPnL[0],bins=75)
np.std(vPnL)


# In[11]:


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


# In[12]:


vPnL_bs=(V_t-np.maximum(S_T-K,0))/V_0


# In[13]:


vPnL_bs=vPnL_bs.reshape(1,-1)
plt.style.use('seaborn-deep')
plt.hist(vPnL_bs[0],bins=75)
np.std(vPnL_bs)


# In[14]:


plt.hist([vPnL_bs[0], vPnL[0]], bins=75, label=['Black-Scholes', 'LSMC'])
plt.legend(loc='upper right')
plt.savefig("Ridge vs Black-Scholes")
plt.show()

