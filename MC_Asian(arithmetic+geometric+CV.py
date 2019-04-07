#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:52:35 2019

@author: Mishty
"""
import numpy as np
from scipy.stats import norm

class MCForAsianOPtions(object):
    
    def __init__(self, S0, K, T, r, n_steps, sigma, option_type):
        
        self.s0 = float(S0)
        self.K = float(K)
        self.T= float(T)
        self.r = float(r)
        self.n_steps = int(n_steps)
        self.sigma = float(sigma)
        self.option_type = option_type
        self.n_simulations = 100000
        self.dt = T/float(n_steps) #time step
        self.df = np.exp(-self.r * self.T) #discount factor
        
    @property
    def simulated_path(self, generator_seed = 51):
        np.random.seed(generator_seed)
#        drift = np.exp(self.r - 0.5*self.sigma**2)*self.dt
#        randn = np.random.randn(self.n_simulations, self.n_steps)
        s_path = (self.s0 *
                      np.cumprod (np.exp ((self.r - 0.5 * self.sigma ** 2) * self.dt) *(np.exp(self.sigma * np.sqrt(self.dt) * np.random.randn(self.n_simulations, self.n_steps))), 1))
#        (self.s0 * np.cumprod(drift*(np.exp(self.sigma * np.sqrt(self.dt)*randn)), 1))
        return s_path
    
    @property
    def payoff_arithmetic(self):
        if self.option_type == 'call':
            payoff_arithmetic = np.exp(-self.r * self.T) \
                         * np.maximum(np.mean(self.simulated_path, 1) - self.K, 0)      
        if self.option_type == 'put':
            payoff_arithmetic = np.maximum(self.K - np.mean(self.simulated_path, 1), 0)*np.exp(-self.r * self.T)
        return payoff_arithmetic
    
    @property
    def payoff_geometric(self):
        if self.option_type == 'call':
            payoff_geometric = np.maximum(np.exp((1/float(self.n_steps)) * np.sum(np.log(self.simulated_path), 1)) - self.K, 0) * np.exp(-self.r * self.T)
        
        if self.option_type == 'put':
            payoff_geometric = np.maximum(self.K - np.exp((1/float(self.n_steps)) * np.sum(np.log(self.simulated_path), 1)), 0) * np.exp(-self.r * self.T)
        return payoff_geometric
    
    @property
    def arithmetic_standardMC(self):
        price_mean = np.mean(self.payoff_arithmetic)
#        price_std = np.std(self.payoff_arithmetic)
#        confmc = [price_mean + 1.96 * price_std/np.sqrt(self.n_simulations), price_mean - 1.96*price_std/np.sqrt(self.n_simulations)] 
#        print(confmc)
        return price_mean
    
    @property
    def geometric_standardMC(self):
        price_mean = np.mean(self.payoff_geometric)
#        price_std = np.std(self.payoff_geometric)
#        confmc = [price_mean + 1.96 * price_std/np.sqrt(self.n_simulations), price_mean - 1.96*price_std/np.sqrt(self.n_simulations)]  
#        print(confmc)
        return price_mean
    
    @property
    def geometric_exact(self):
        sigsqT = (self.sigma ** 2 * self.T * (self.n_steps + 1) * (2 * self.n_steps +1))/(6 * (self.n_steps ** 2))
        muT = (0.5 * sigsqT + (self.r - 0.5 * self.sigma **2) * self.T * (self.n_steps + 1)/(2 * self.n_steps))
        
        d1 = ((np.log(self.s0 / self.K) + (muT + 0.5 * sigsqT)) / np.sqrt(sigsqT))
        d2 = d1 - np.sqrt(sigsqT)
        if self.option_type == 'call':
            N1 = norm.cdf(d1)
            N2 = norm.cdf(d2)
            geometric_price = np.exp(-self.r * self.T) * (self.s0 * np.exp(muT) * N1 - self.K * N2)
        if self.option_type == 'put':
            N1 = norm.cdf(-d1)
            N2 = norm.cdf(-d2)
            geometric_price = np.exp(-self.r * self.T) * (self.K * N2 - self.s0 * np.exp(muT) * N1)
        return geometric_price
    
    @property
    def calculate_theta(self):
        covXY = np.mean(self.payoff_arithmetic * self.payoff_geometric) - np.mean(self.payoff_arithmetic )*np.mean(self.payoff_geometric)
        theta = covXY/np.var(self.payoff_geometric)
        return theta
    
    @property
    def control_variate(self):
        Z = self.payoff_arithmetic + self.calculate_theta * (self.geometric_exact - self.payoff_geometric)
        Zmean = np.mean(Z, 0)
#        confcv = [Zmean - 1.96 * np.std(Z)/np.sqrt(self.n_simulations), Zmean + 1.96*np.std(Z)/np.sqrt(self.n_simulations)]
        return Zmean
    

asian_call = MCForAsianOPtions(100.0, 100.0, 3.0, 0.05, 50, 0.3, 'call')
print('\nCall Options')
print(asian_call.arithmetic_standardMC)
print(asian_call.geometric_standardMC)
print(asian_call.control_variate)

print('\nPut Options')
asian_put = MCForAsianOPtions(100.0, 100.0, 3.0, 0.05, 50, 0.3, 'put')
print(asian_put.arithmetic_standardMC)
print(asian_put.geometric_standardMC)
print(asian_put.control_variate)
    
    
        

