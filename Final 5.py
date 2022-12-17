#!/usr/bin/env python
# coding: utf-8

# In[18]:


#Packages
import torch
from torch.distributions.normal import Normal
from torch.nn import Linear
from torch.nn import ReLU
import typing
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn.functional as fn
import numpy.random as npr
from torch.optim import Adam
from tqdm import tqdm
from scipy.stats import norm


# In[19]:


def european_option_delta(log_moneyness, time_expiry, volatility) -> torch.Tensor:
    
    s, t, v = map(torch.as_tensor, (log_moneyness, time_expiry, volatility)) #convert to tensor
    normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
    return normal.cdf((s + (v ** 2 / 2) * t) / (v * torch.sqrt(t)))





class MultiLayerPerceptron(torch.nn.ModuleList):
   

    
    def __init__(self, in_features, out_features, n_layers=4, n_units=52):
        super().__init__()
        for n in range(n_layers):
            i = in_features if n == 0 else n_units
            self.append(Linear(i, n_units))
            self.append(ReLU())
        self.append(Linear(n_units, out_features)) #notice out of loop so that the loop only passes through the layer part

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


def generate_geometric_brownian_motion(
    n_paths, maturity=30 / 365, dt=1 / 365, volatility=0.2, device=None
) -> torch.Tensor:
    
    randn = torch.randn((int(maturity / dt), n_paths), device=device)
    randn[0, :] = 0.0
    bm = volatility * (dt ** 0.5) * randn.cumsum(0)
    t = torch.linspace(0, maturity, int(maturity / dt))[:, None].to(bm)
    return torch.exp(bm - (volatility ** 2) * t / 2)


def entropic_loss(pnl) -> torch.Tensor:
    
    return -torch.mean(-torch.exp(-pnl))


def to_premium(pnl) -> torch.Tensor:
    
 
    return -torch.log(entropic_loss(pnl))


# In[20]:


#Whenever it is avaialble, use the graphic process
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# In[21]:


# In each epoch, N_PATHS brownian motion time-series are generated.
#N_PATHS = 5000
# How many times a model is updated in the experiment.
N_EPOCHS = 400

#I am using small number of epochs because the computer complained, due to memory issues.


# In[22]:


def to_numpy(tensor: torch.Tensor) -> np.array:
    return tensor.cpu().detach().numpy()


# In[23]:


def european_option_payoff(prices: torch.Tensor, strike=1.0) -> torch.Tensor:
   
    return fn.relu(prices[-1, :] - strike)


# In[24]:


#Defined this payoff function for 1 dimensinal vector.

def european_option_payoff_forward(prices: torch.Tensor, strike=1.0) -> torch.Tensor:

    
    return fn.relu(prices[-1] - strike)


# In[25]:


T=1
n=1000
K=1
sigma=0.2
r=0
S0=1
S_T=S0
steps=52
dt=T/steps
vol0=0.2
vol=0.5


# In[26]:


#Gnerate the data
np.random.seed(123)
price_vec=[]


for i in range(1,steps):
    T_=T-(i-1)*dt

    vS0=S0*np.exp(-0.5*vol0*vol0*dt + vol0*np.sqrt(T)*npr.normal(0,1,(n,1)))
    vS_T=vS0*np.exp(-0.5*vol*vol*dt + vol*np.sqrt(T-(i-1)*dt)*npr.normal(0,1,(n,1)))
    vC=np.maximum(vS_T-K,0)
    price_vec.append(vS_T)
    


# In[27]:


#Gnerate the data
np.random.seed(123)
price_vec=[]


for i in range(1,steps):
    T_=T-(i-1)*dt

    vS0=S0*np.exp(-0.5*vol*vol*dt + vol0*np.sqrt(T)*npr.normal(0,1,(n,1)))
    vS_T=vS0*np.exp(-0.5*vol*vol*dt + vol*np.sqrt(T-(i-1)*dt)*npr.normal(0,1,(n,1)))
    vC=np.maximum(vS_T-K,0)
    price_vec.append(vS_T)
    


# In[28]:


#Convert it to tensor form
#generate_geometric_brownian_motion(n_paths=N_PATHS, maturity=maturity, dt=dt, volatility=volatility, device=device)
price_vec=np.array(price_vec)
price_vec=price_vec.reshape(steps-1,n)
vS0=vS0.reshape(1,-1)
price_vec=np.concatenate((vS0,price_vec))
price_vec=torch.from_numpy(price_vec).float()


# In[29]:


#Parameters for the training set

N_Paths=n
N_PATHS=N_Paths,
maturity=T
dt=1 / 52
volatility=0.2
cost: float


# In[30]:


def compute_profit_and_loss(
    hedging_model: torch.nn.Module,
    payoff: typing.Callable[[torch.Tensor], torch.Tensor],
    prices:typing.Callable[[torch.Tensor], torch.Tensor],
    cost: float,
    n_paths=N_PATHS,
    maturity=1,
    dt=1 / 52,
    volatility=0.2,
    
) -> torch.Tensor:
 
    
    # Prepare time-series of prices with shape (time, batch)
    #prices = generate_geometric_brownian_motion(
        #n_paths, maturity=maturity, dt=dt, volatility=volatility, device=device
    #)
    
    #write a if statement
    hedge = torch.zeros_like(prices[:1]).reshape(-1)   
    pnl = 0
    # Simulate hedging over time.
    for n in range(prices.shape[0] - 1):
        # Prepare a model input.
        #x_log_moneyness = prices[n, :, None].log()
        
        if prices.size(-1)==N_Paths:
            x_log_moneyness = prices[n, :, None].log()
        else:
            x_log_moneyness = prices[n,None].log()
        x_time_expiry = torch.full_like(x_log_moneyness, maturity - n * dt) #vector of the same size of x_log_moneyness with maturity...
        x_volatility = torch.full_like(x_log_moneyness, volatility)
        #x = torch.cat([x_log_moneyness, x_time_expiry, x_volatility], 1)
        if prices.size(-1)==N_Paths:
            x = torch.cat([x_log_moneyness, x_time_expiry, x_volatility], 1)
        else:
            x = torch.cat([x_log_moneyness, x_time_expiry, x_volatility],0 )
        

        # Infer a preferable hedge ratio.
        prev_hedge = hedge
        hedge = hedging_model(x)

        # Receive profit/loss from the original asset.
        pnl += hedge * (prices[n + 1] - prices[n])
        # Pay transaction cost.
        pnl -= cost * torch.abs(hedge - prev_hedge) * prices[n]

    # Pay the option's payoff to the customer.
    pnl -= payoff(prices)

    return pnl


# In[31]:


#Define the model as the neural network defined a priori. Other architectures can be tried.
model_ntb= MultiLayerPerceptron(3,1).to(device)


# In[32]:


#Apply the pnl fucntion to the later so that hedging weights are given by deep neural network.
torch.manual_seed(42)#seed should be above, before generatinf the data.
pnl_ntb = compute_profit_and_loss(model_ntb, european_option_payoff,prices=price_vec, cost=0)


# In[33]:


#Defining how the weights are going to be computed.
#Specifing the loss as in Buelher(2020)


def fit(
    hedging_model: torch.nn.Module,
    payoff: typing.Callable[[torch.Tensor], torch.Tensor],
    prices:typing.Callable[[torch.Tensor], torch.Tensor],
    cost: float,
    n_epochs=N_EPOCHS,
) -> list:
    
    optim = Adam(hedging_model.parameters())

    loss_history = []
    progress = tqdm(range(n_epochs))

    for _ in progress:
        optim.zero_grad()
        pnl = compute_profit_and_loss(hedging_model, payoff,prices=prices, cost=cost)
        loss = entropic_loss(pnl)
        loss.backward()
        optim.step()

        progress.desc = f"Loss={loss:.5f}"
        loss_history.append(loss.item())

    return loss_history


# In[34]:


#Training the model
torch.manual_seed(42)
history_ntb = fit(model_ntb, european_option_payoff,prices=price_vec, cost=0)


# In[35]:


#This function is good for Monte Carlo, once it allows to compute the mean given a sample of data for each time period.
#However it does not work with 1 dimensional vector


def evaluate_premium(
    hedging_model: torch.nn.Module,
    payoff: typing.Callable[[torch.Tensor], torch.Tensor],
    prices:typing.Callable[[torch.Tensor], torch.Tensor],
    cost: float,
    n_times=20,
) -> float:
   
    with torch.no_grad():
        p = lambda: to_premium(
            compute_profit_and_loss(hedging_model, payoff,prices ,cost=0)
        ).item()
        return float(np.mean([p() for _ in range(n_times)]))


# In[36]:


#Price for the data used to train the model.

torch.manual_seed(42)
premium_ntb = evaluate_premium(model_ntb, european_option_payoff,prices=price_vec, cost=0)

premium_ntb #0.02259032055735588 which is a sensible value


# In[37]:


def evaluate_pnl(
    hedging_model: torch.nn.Module,
    payoff: typing.Callable[[torch.Tensor], torch.Tensor],
    prices:typing.Callable[[torch.Tensor], torch.Tensor],
    cost: float,
) -> float:
   
    with torch.no_grad():
        p =-compute_profit_and_loss(hedging_model,european_option_payoff,prices, cost=0).item()
        return p


# In[54]:


np.random.seed(123)
price_test=[]

for i in range(steps):
    S_T=S_T*np.exp(-0.5*sigma*sigma*dt+np.sqrt(dt)*sigma*npr.normal(0,1,(n,1)))
    price_test.append(S_T)


# In[55]:


price_test=np.array(price_test)
price_test=price_test.reshape(steps,n)
price_test=torch.from_numpy(price_test).float()




# In[40]:


pnl_deep=[]
for i in range(n):
    with torch.no_grad():
        p=-compute_profit_and_loss(model_ntb,european_option_payoff_forward,prices=price_test[:,i], cost=0)
        p1=p.item()
        pnl_deep.append(p1)
        
   


# In[41]:



pnl_deep=np.array(pnl_deep)


# In[42]:


plt.style.use('seaborn-deep')
plt.hist(pnl_deep,bins=75)
np.std(pnl_deep)


# In[ ]:


def BlackScholes(S0,r,sigma,T,K):
    d1 =  1 / (sigma * np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    d2 = d1 - sigma * np.sqrt(T)
    return norm.cdf(d1) * S0 - norm.cdf(d2) * K * np.exp(-r*T)

def BlackScholesCallDelta(S0,r,sigma,T,K):
    d1 =  1 / (sigma * np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    return norm.cdf(d1)

BlackScholes_vec=np.vectorize(BlackScholes)

BlackScholesCallDelta_vec=np.vectorize(BlackScholesCallDelta)


# In[56]:


T=1
n=1000
K=1
sigma=0.2
r=0

steps=52
dt=T/steps


# In[57]:


V_0_bs=BlackScholes_vec(S0,r,sigma,T,K)
a_t_bs=BlackScholesCallDelta(S0,r,sigma,T,K)
b_t_bs=V_0_bs-a_t_bs*S0_bs

np.random.seed(123)
#V_t=[]
for i in range(steps):
    S_T_bs=S_T_bs*np.exp(-0.5*sigma*sigma*dt+np.sqrt(dt)*sigma*npr.normal(0,1,(n,1)))
    #S_T=S_T*np.exp(0.1*dt+np.sqrt(dt)*sigma*npr.normal(0,1,(n,1)))
    V_t_bs=a_t_bs*S_T_bs+b_t_bs
    a_t_bs=BlackScholesCallDelta(S_T_bs,r,sigma,T-i*dt,K)
    b_t_bs=V_t_bs-a_t_bs*S_T_bs


# In[58]:


vPnL_bs=(V_t_bs-np.maximum(S_T_bs-K,0))/V_0_bs


# In[59]:


vPnL_bs=vPnL_bs.reshape(1,-1)
plt.style.use('seaborn-deep')
plt.hist(vPnL_bs[0],bins=75)
np.std(vPnL_bs)


# In[83]:


plt.hist([vPnL_bs[0],pnl_deep], bins=75, label=['Black-Scholes', 'Deep Hedging'])
plt.legend(loc='upper right')
plt.savefig("Deep_hedging vs BS")
plt.show()

