#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
from scipy.stats import norm
from numpy import linalg as la
import numpy.random as npr
from tabulate import tabulate
from matplotlib import pyplot as plt
import random
import math
from numpy.fft import *


# In[2]:


def M76_characteristic_function(u, x0, T, r, sigma, lamb, mu, delta):
    ''' Valuation of European call option in M76 model via
    Lewis (2001) Fourier-based approach: characteristic function.
    Parameter definitions see function M76_value_call_INT. '''
    omega = x0 / T + r - 0.5 * sigma ** 2                 - lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
    value = np.exp((1j * u * omega - 0.5 * u ** 2 * sigma ** 2 +
            lamb * (np.exp(1j * u * mu - u ** 2 * delta ** 2 * 0.5) - 1))  * T)
    return value

#
# Valuation by FFT
#


def M76_value_call_FFT(S0, K, T, r, sigma, lamb, mu, delta):
    ''' Valuation of European call option in M76 model via
    Carr-Madan (1999) Fourier-based approach.
    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump
    Returns
    =======
    call_value: float
        European call option present value
    '''
    k = math.log(K / S0)
    x0 = math.log(S0 / S0)
    g = 2  # factor to increase accuracy
    N = g * 4096
    eps = (g * 150.) ** -1
    eta = 2 * math.pi / (N * eps)
    b = 0.5 * N * eps - k
    u = np.arange(1, N + 1, 1)
    vo = eta * (u - 1)
    # Modificatons to Ensure Integrability
    if S0 >= 0.95 * K:  # ITM case
        alpha = 1.5
        v = vo - (alpha + 1) * 1j
        mod_char_fun = math.exp(-r * T) * M76_characteristic_function(
                                    v, x0, T, r, sigma, lamb, mu, delta) \
                / (alpha ** 2 + alpha - vo ** 2 + 1j * (2 * alpha + 1) * vo)
    else:  # OTM case
        alpha = 1.1
        v = (vo - 1j * alpha) - 1j
        mod_char_fun_1 = math.exp(-r * T) * (1 / (1 + 1j * (vo - 1j * alpha))
                                   - math.exp(r * T) / (1j * (vo - 1j * alpha))
                                   - M76_characteristic_function(
                                     v, x0, T, r, sigma, lamb, mu, delta)
                / ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha)))
        v = (vo + 1j * alpha) - 1j
        mod_char_fun_2 = math.exp(-r * T) * (1 / (1 + 1j * (vo + 1j * alpha))
                                   - math.exp(r * T) / (1j * (vo + 1j * alpha))
                                   - M76_characteristic_function(
                                     v, x0, T, r, sigma, lamb, mu, delta)
                / ((vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha)))
    
    # Numerical FFT Routine
    delt = np.zeros(N, dtype=np.float)
    delt[0] = 1
    j = np.arange(1, N + 1, 1)
    SimpsonW = (3 + (-1) ** j - delt) / 3
    if S0 >= 0.95 * K:
        fft_func = np.exp(1j * b * vo) * mod_char_fun * eta * SimpsonW
        payoff = (fft(fft_func)).real
        call_value_m = np.exp(-alpha * k) / math.pi * payoff
    else:
        fft_func = (np.exp(1j * b * vo)
                    * (mod_char_fun_1 - mod_char_fun_2)
                    * 0.5 * eta * SimpsonW)
        payoff = (fft(fft_func)).real
        call_value_m = payoff / (np.sinh(alpha * k) * math.pi)
    pos = int((k + b) / eps)
    call_value = call_value_m[pos]
    return call_value * S0



# In[3]:


M76_value_call_FFT=np.vectorize(M76_value_call_FFT)


# In[4]:


def jump_diffusion(S=1, X=0.5, T=1, mu=0.12, sigma=0.3, Lambda=0.25,
                   a=0.2, b=0.2, Nsteps=252, Nsim=100, alpha=0.05, seed=None):
    
    # Import required libraries
    import time
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set random seed
    np.random.seed(seed)

    '''
    Time the whole path-generating process, using a tic-toc method familiar
    to MATLAB users
    '''
    tic = time.time()

    # Calculate the length of the time step
    Delta_t = T/Nsteps

    '''
    Compute mean and variance of a standard lognormal distribution from user
    defined parameters a and b. The latter are useful to simulate the jump
    component in Monte Carlo.
    a and b are chosen such that log(Y(j)) ~ N(a, b**2). This implies that the
    mean and variance of the multiplicative jumps will be:
     * mean_Y = np.exp(a + 0.5*(b**2))
     * variance_Y = np.exp(2*a + b**2) * (np.exp(b**2)-1)
    '''
    mean_Y = np.exp(a + 0.5*(b**2))
    variance_Y = np.exp(2*a + b**2) * (np.exp(b**2)-1)

    '''
    Calculate the theoretical drift (M) and volatility (V) of the stock price
    process under Merton's jump diffusion model. These values can be used to
    monitor the rate of convergence of Monte Carlo estimates as the number of
    simulated experiments increases, and can help spot errors, if any, in
    implementing the model.
    '''
    M = S * np.exp(mu*T + Lambda*T*(mean_Y-1))
    V = S**2 * (np.exp((2*mu + sigma**2)*T         + Lambda*T*(variance_Y + mean_Y**2 - 1))         - np.exp(2*mu*T + 2*Lambda*T*(mean_Y - 1)))

    '''
    Generate an Nsim x (Nsteps+1) array of zeros to preallocate the simulated
    paths of the Monte Carlo simulation. Each row of the matrix represents a
    full, possible path for the stock, each column all values of the asset at
    a particular instant in time.
    '''
    simulated_paths = np.zeros([Nsim, Nsteps+1])

    # Replace the first column of the array with the vector of initial price S
    simulated_paths[:,0] = S

    '''
    To account for the multiple sources of uncertainty in the jump diffusion
    process, generate three arrays of random variables.
     - The first one is related to the standard Brownian motion, the component
       epsilon(0,1) in epsilon(0,1) * np.sqrt(dt);
     - The second and third ones model the jump, a compound Poisson process:
       the former (a Poisson process with intensity Lambda) causes the asset
       price to jump randomly (random timing); the latter (a Gaussian variable)
       defines both the direction (sign) and intensity (magnitude) of the jump.
    '''
    Z_1 = np.random.normal(size=[Nsim, Nsteps])
    Z_2 = np.random.normal(size=[Nsim, Nsteps])
    Poisson = np.random.poisson(Lambda*Delta_t, [Nsim, Nsteps])

    # Populate the matrix with Nsim randomly generated paths of length Nsteps
    for i in range(Nsteps):
        simulated_paths[:,i+1] = simulated_paths[:,i]*np.exp((mu
                               - sigma**2/2)*Delta_t + sigma*np.sqrt(Delta_t) \
                               * Z_1[:,i] + a*Poisson[:,i] \
                               + np.sqrt(b**2) * np.sqrt(Poisson[:,i]) \
                               * Z_2[:,i])

    # Single out array of simulated prices at maturity T
    final_prices = simulated_paths[:,-1]

    # Compute mean, variance, standard deviation, skewness, excess kurtosis
    mean_jump = np.mean(final_prices)
    var_jump = np.var(final_prices)
    std_jump = np.std(final_prices)
    skew_jump = stats.skew(final_prices)
    kurt_jump = stats.kurtosis(final_prices)

    # Calculate confidence interval for the mean
    ci_low = mean_jump - std_jump/np.sqrt(Nsim)*stats.norm.ppf(1-0.5*alpha)
    ci_high = mean_jump + std_jump/np.sqrt(Nsim)*stats.norm.ppf(1-0.5*alpha)

    
    # Generate t, the time variable on the abscissae
    t = np.linspace(0, T, Nsteps+1) * Nsteps
    # Time and print the elapsed time
    toc = time.time()
    elapsed_time = toc - tic
    #print('Total running time: {:.2f} ms'.format(elapsed_time*1000))
    
    return simulated_paths


# In[17]:


S0=1
K=1
T=1
r=0
sigma=0.2
lamb=0.25
mu=0.12
steps=52
delta=0.5
dt=T/steps
c=[1,2]


# In[11]:


V_0=M76_value_call_FFT(S0, K, T, r, sigma, lamb, mu, delta)
a_t=((M76_value_call_FFT(S0, K, T, r, sigma, lamb, mu, delta)-M76_value_call_FFT(prices_test[0,1], K, T, r, sigma, lamb, mu, delta)))/(S0-prices_test[0,1])
b_t=V_0-a_t*S0


np.random.seed(123)

for i in range(1,steps-1):
   
    S_T=prices_test[:,i]
    V_t=a_t*S_T+b_t
    a_t=((M76_value_call_FFT(prices_test[:,i+1], K,T-i*dt , r, sigma, lamb, mu, delta)-M76_value_call_FFT(prices_test[:,i], K, T, r, sigma, lamb, mu, delta)))/(prices_test[:,i+1]-prices_test[:,i])
    b_t=V_t-a_t*S_T


# In[12]:


vPnL=(V_t-np.maximum(S_T-K,0))/V_0
 


# In[1]:


vPnL=vPnL.reshape(1,-1)
np.std(vPnL)

