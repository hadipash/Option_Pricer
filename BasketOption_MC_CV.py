# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:59:36 2019

@author: user
"""

import numpy as np
from scipy.stats import norm


class BasketOption:
    def __init__(self, S1, S2, K, r, T, σ1, σ2, ρ, n_steps,option_type, m=int(1e4), t=0.0):           #Added
        self.__S1, self.__S2 = S1, S2
        self.__K, self.__r = K, r
        self.__σ1, self.__σ2 = σ1, σ2
        self.__ρ, self.__Δ = ρ, T - t
        
        self.option_type = option_type                                       #Added
        self.n_steps = int(n_steps)                                          #Added
        self.n_simulations = 100000                                        #Added
        self.dt = T/float(n_steps) #time step                                  #Added
        self.df = np.exp(-self.__r * self.__Δ) #discount factor                #Added

        self.__B0__Geo = self.__geo_mean([self.__S1, self.__S2], axis=0)
        self.__B0__Ari = np.mean([self.__S1, self.__S2])   #Added
        σB_sqΔ = self.__Δ * (self.__σ1 ** 2 + 2 * self.__ρ * self.__σ1 * self.__σ2 + self.__σ2 ** 2) / 4
        self.σB_sqΔ=σB_sqΔ
        self.σB_sqΔ_Ari =np.sqrt(self.__σ1**2+self.__σ2**2+2*self.__σ2*self.__σ1*self.__ρ)/2               #Added
        
        #Variance = (w(1)^2 x o(1)^2) + (w(2)^2 x o(2)^2) + (2 x (w(1)o(1)w(2)o(2)q(1,2))
        
        self.__μBΔ = self.__Δ * (self.__r - (self.__σ1 ** 2 + self.__σ2 ** 2) / 4) + σB_sqΔ / 2

        self.__d1hat = (np.log(self.__B0__Geo / self.__K) + self.__μBΔ + σB_sqΔ / 2) / np.sqrt(σB_sqΔ)
        self.__d2hat = self.__d1hat - np.sqrt(σB_sqΔ)


    @property                                                                #Added
    def simulated_path_Ari(self, generator_seed = 51):
        np.random.seed(generator_seed)
#        drift = np.exp(self.__r - 0.5*self.sigma**2)*self.dt
#        randn = np.random.randn(self.n_simulations, self.n_steps)
        s_path = (self.__B0__Ari *
                      np.cumprod (np.exp ((self.__r - 0.5 * self.σB_sqΔ_Ari ** 2) * self.dt) *(np.exp(self.σB_sqΔ_Ari * np.sqrt(self.dt) * np.random.randn(self.n_simulations, self.n_steps))), 1))
#        (self.s0 * np.cumprod(drift*(np.exp(self.sigma * np.sqrt(self.dt)*randn)), 1))
        return s_path
    
    @property                                                                #Added
    def simulated_path_Geo(self, generator_seed = 75):
        np.random.seed(generator_seed)
#        drift = np.exp(self.__r - 0.5*self.sigma**2)*self.dt
#        randn = np.random.randn(self.n_simulations, self.n_steps)
        s_path = (self.__B0__Geo *
                      np.cumprod (np.exp ((self.__r - 0.5 * self.σB_sqΔ ** 2) * self.dt) *(np.exp(self.σB_sqΔ * np.sqrt(self.dt) * np.random.randn(self.n_simulations, self.n_steps))), 1))
#        (self.s0 * np.cumprod(drift*(np.exp(self.sigma * np.sqrt(self.dt)*randn)), 1))
        return s_path 
    
    
    @property
    def payoff_arithmetic(self):
        if self.option_type == 'call':
            payoff_arithmetic = np.exp(-self.__r *self.__Δ ) \
                         * np.maximum(self.simulated_path_Ari[:,-1] - self.__K, 0)     
            
        if self.option_type == 'put':
            payoff_arithmetic = np.maximum(self.__K - self.simulated_path_Ari[:,-1], 0)*np.exp(-self.__r * self.__Δ )
            
        return payoff_arithmetic
    
    @property
    def payoff_geometric(self):
        if self.option_type == 'call':
            payoff_geometric = np.exp(-self.__r *self.__Δ ) \
                         * np.maximum(self.simulated_path_Geo[:,-1] - self.__K, 0)     
            
        if self.option_type == 'put':
            payoff_geometric = np.maximum(self.__K - self.simulated_path_Geo[:,-1], 0)*np.exp(-self.__r * self.__Δ )
        return payoff_geometric
    
    
    
    
    @property                                                                #Added
    def arithmetic_standardMC(self):
        price_mean = np.mean(self.payoff_arithmetic)
#        price_std = np.std(self.payoff_arithmetic)
#        confmc = [price_mean + 1.96 * price_std/np.sqrt(self.n_simulations), price_mean - 1.96*price_std/np.sqrt(self.n_simulations)] 
#        print(confmc)
        return price_mean
    
    @property                                                                #Added
    def geometric_standardMC(self):
        price_mean = np.mean(self.payoff_geometric)
#        price_std = np.std(self.payoff_geometric)
#        confmc = [price_mean + 1.96 * price_std/np.sqrt(self.n_simulations), price_mean - 1.96*price_std/np.sqrt(self.n_simulations)]  
#        print(confmc)
        return price_mean
    
    
    @staticmethod
    def __geo_mean(x, axis=1):
        return np.exp(np.sum(np.log(x), axis=axis) / len(x))
    


    def geo_call_cf(self):
        return np.exp(-self.__r * self.__Δ) * (self.__B0__Geo * np.exp(self.__μBΔ) * norm.cdf(self.__d1hat) -
                                               self.__K * norm.cdf(self.__d2hat))

    def geo_put_cf(self):
        return np.exp(-self.__r * self.__Δ) * (self.__K * norm.cdf(-self.__d2hat) -
                                               self.__B0__Geo * np.exp(self.__μBΔ) * norm.cdf(-self.__d1hat))
        
    @property
    def geometric_exact(self):

        if self.option_type == 'call':
            geometric_price = self.geo_call_cf
        if self.option_type == 'put':
            geometric_price = self.geo_put_cf
        return geometric_price
    
    def test(self):
        return self.σB_sqΔ_Ari 
    
    @property
    def calculate_theta(self):
        covXY = np.mean(self.payoff_arithmetic * self.payoff_geometric) - np.mean(self.payoff_arithmetic )*np.mean(self.payoff_geometric)
        theta = covXY/np.var(self.payoff_geometric)
        return theta
    
    @property
    def control_variate(self):
        Z = self.payoff_arithmetic + self.calculate_theta * (self.geometric_exact() - self.payoff_geometric)
        Zmean = np.mean(Z, 0)
#        confcv = [Zmean - 1.96 * np.std(Z)/np.sqrt(self.n_simulations), Zmean + 1.96*np.std(Z)/np.sqrt(self.n_simulations)]
        return Zmean

# test, closed-form formula
bo = BasketOption(S1=100, S2=100, K=100, r=0.1, T=3, σ1=0.3, σ2=0.3, ρ=0.5,n_steps=5,option_type='call')
print("Should be like this, will double check tmr:")
#print(bo.geometric_exact())
#print(bo.payoff_geometric)
#print(bo.payoff_arithmetic)
#print(bo.calculate_theta)
#print(bo.geometric_exact())
print(bo.arithmetic_standardMC)
print(bo.control_variate)

print("Should be like this, will double check tmr:")
#print(bo.geometric_exact())
#print(bo.payoff_geometric)
#print(bo.payoff_arithmetic)
#print(bo.calculate_theta)
#print(bo.geometric_exact())
print(bo.geometric_exact())
print(bo.geometric_standardMC)