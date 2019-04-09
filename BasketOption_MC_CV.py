# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 00:54:11 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:59:36 2019
@author: user
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


class BasketOption:
    def __init__(self, S1, S2, K, r, T, σ1, σ2, ρ, n_steps,option_type, m=int(1e4), t=0.0):           #Added
        self.__S1, self.__S2 = S1, S2
        self.__K, self.__r = K, r
        self.__σ1, self.__σ2 = σ1, σ2
        self.__ρ, self.__Δ = ρ, T - t
        
        self.option_type = option_type                                       #Added
        self.n_steps = int(n_steps)                                          #Added
        self.n_simulations =  100000                                      #Added
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
        self.__S1_path_a = []
        self.__S2_path_a = []
        self.__S1_path_g = []
        self.__S2_path_g = []
        
    @property                                                                #Added
    def simulated_path_Ari(self, generator_seed = 52):
        self.__S1_path_a = []
        self.__S2_path_a = []
        np.random.seed(generator_seed)
        drift = np.exp((self.__r - 0.5*self.__σ1**2)*self.dt)
        drift2 = np.exp((self.__r - 0.5*self.__σ2**2)*self.dt)
        randn1 = np.random.randn(self.n_simulations, self.n_steps)
        
        #self.__S1_path = (self.__S1 *
                    #np.cumprod (np.exp ((self.__r - 0.5 * self.__σ1 ** 2) * self.dt) *(np.exp(self.__σ1 * np.sqrt(self.dt) * randn1)), 1))
        gf1 = drift * np.exp(self.__σ1 * np.sqrt(self.dt) * randn1)
        
        np.random.seed(generator_seed-10)
        randn2 = np.random.randn(self.n_simulations, self.n_steps)    
        randn3 = np.add(self.__ρ * randn1 , np.sqrt(1-self.__ρ**2) * randn2)
        gf2 = drift2 * np.exp(self.__σ2 * np.sqrt(self.dt) * randn3)
        #self.__S2_path = (self.__S2 *
                    #np.cumprod (np.exp ((self.__r - 0.5 * self.__σ2 ** 2) * self.dt) *(np.exp(self.__σ2 * np.sqrt(self.dt) * randn3)), 1))      
        
        for i in range(self.n_simulations):
            self.__S1_path_a.append([self.__S1 * gf1[i][0]]) # time step =1  OR  i = 0 for all M paths : S(1) = S(0) * exp(rdt - sigma^2 *dt /2) * exp(sigma *sqrt(dt) * dZ)
            for j in range(1, self.n_steps): #N-1 time steps here
                self.__S1_path_a[i].append(self.__S1_path_a[i][j - 1] * gf1[i][j]) #generate Whole MC , S(2) to S(N)
            
            
        for i in range(self.n_simulations):
            self.__S2_path_a.append([self.__S2 * gf2[i][0]]) # time step =1  OR  i = 0 for all M paths : S(1) = S(0) * exp(rdt - sigma^2 *dt /2) * exp(sigma *sqrt(dt) * dZ)
            for j in range(1, self.n_steps): #N-1 time steps here
                self.__S2_path_a[i].append(self.__S2_path_a[i][j - 1] * gf2[i][j]) #generate Whole MC , S(2) to S(N)           
            

    
    
        s_path = np.add(self.__S1_path_a,self.__S2_path_a)/2
        
        #s_path= [(e+d)/2 for e,d in self.__S1_path_a,self.__S2_path_a]
        
        return s_path
    
    
    @property                                                                #Added
    def simulated_path_Geo(self, generator_seed = 51):
        np.random.seed(generator_seed)
#        drift = np.exp(self.__r - 0.5*self.sigma**2)*self.dt
#        randn = np.random.randn(self.n_simulations, self.n_steps)
        s_path = (self.__B0__Geo *
                      np.cumprod (np.exp ((self.__μBΔ - 0.5 * self.σB_sqΔ ** 2) * self.dt) *(np.exp(self.σB_sqΔ * np.sqrt(self.dt) * np.random.randn(self.n_simulations, self.n_steps))), 1))
#        (self.s0 * np.cumprod(drift*(np.exp(self.sigma * np.sqrt(self.dt)*randn)), 1))
        return s_path 
    


    @property                                                                #Added
    def simulated_path_Geo2(self, generator_seed = 52):
        self.__S1_path_g = []
        self.__S2_path_g = []
        np.random.seed(generator_seed)
        drift = np.exp((self.__r - 0.5*self.__σ1**2)*self.dt)
        drift2 = np.exp((self.__r - 0.5*self.__σ2**2)*self.dt)
        randn1 = np.random.randn(self.n_simulations, self.n_steps)
        gf1 = drift * np.exp(self.__σ1 * np.sqrt(self.dt) * randn1)
        
        np.random.seed(generator_seed-10)
        randn2 = np.random.randn(self.n_simulations, self.n_steps)    
        randn3 = np.add(self.__ρ * randn1 , np.sqrt(1-self.__ρ**2) * randn2)
        gf2 = drift2 * np.exp(self.__σ2 * np.sqrt(self.dt) * randn3)
        
        
        for i in range(self.n_simulations):
            self.__S1_path_g.append([self.__S1 * gf1[i][0]]) # time step =1  OR  i = 0 for all M paths : S(1) = S(0) * exp(rdt - sigma^2 *dt /2) * exp(sigma *sqrt(dt) * dZ)
            for j in range(1, self.n_steps): #N-1 time steps here
                self.__S1_path_g[i].append(self.__S1_path_g[i][j - 1] * gf1[i][j]) #generate Whole MC , S(2) to S(N)
            
            
        for i in range(self.n_simulations):
            self.__S2_path_g.append([self.__S2 * gf2[i][0]]) # time step =1  OR  i = 0 for all M paths : S(1) = S(0) * exp(rdt - sigma^2 *dt /2) * exp(sigma *sqrt(dt) * dZ)
            for j in range(1, self.n_steps): #N-1 time steps here
                self.__S2_path_g[i].append(self.__S2_path_g[i][j - 1] * gf2[i][j]) #generate Whole MC , S(2) to S(N)           
            
 
        s_path_g = np.sqrt( np.multiply(self.__S1_path_g , self.__S2_path_g) )
        return s_path_g 


    
    @property
    def payoff_arithmetic(self):
        if self.option_type == 'call':
            payoff_arithmetic = np.exp(-self.__r *self.__Δ ) \
                         * np.maximum(self.simulated_path_Ari[:,-1] - self.__K, 0)
            #payoff_arithmetic=1
        if self.option_type == 'put':
            payoff_arithmetic = np.maximum(self.__K - self.simulated_path_Ari[:,-1], 0)*np.exp(-self.__r * self.__Δ )
            #payoff_arithmetic=1
        return payoff_arithmetic
    
    @property
    def payoff_geometric(self):
        if self.option_type == 'call':
            payoff_geometric = np.exp(-self.__r *self.__Δ ) \
                         * np.maximum(self.simulated_path_Geo2[:,-1] - self.__K, 0)     
            
        if self.option_type == 'put':
            payoff_geometric = np.maximum(self.__K - self.simulated_path_Geo2[:,-1], 0)*np.exp(-self.__r * self.__Δ )
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
        return np.sqrt(x[0]*x[1])
    


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
        return self.σB_sqΔ
    
    def test2(self):
        return self.__μBΔ
    
    @property
    def calculate_theta(self):
        covXY = np.mean(self.payoff_arithmetic * self.payoff_geometric) - np.mean(self.payoff_arithmetic )*np.mean(self.payoff_geometric)
        theta = covXY/np.var(self.payoff_geometric)
       # mean=np.mean(self.payoff_arithmetic )
        #g= self.geometric_standardMC
        #theta=[]
        #theta =np.mean(np.multiply(self.payoff_arithmetic , self.payoff_geometric)) 
        return theta
    
    @property
    def control_variate(self):
        Z = self.payoff_arithmetic + self.calculate_theta * (self.geometric_exact() - self.payoff_geometric)
        Zmean = np.mean(Z, 0)
#        confcv = [Zmean - 1.96 * np.std(Z)/np.sqrt(self.n_simulations), Zmean + 1.96*np.std(Z)/np.sqrt(self.n_simulations)]
        return Zmean

# test, closed-form formula
bo = BasketOption(S1=100, S2=100, K=100, r=0.05, T=3, σ1=0.3, σ2=0.3, ρ=1,n_steps=15,option_type='call')

print("Call - geo MC vs  closed form:")
#print(bo.calculate_theta)
print(bo.geometric_standardMC)
print(bo.geometric_exact())


print("Call -arith MC vs Control:")
print(bo.arithmetic_standardMC)
#print(bo.payoff_arithmetic)
##print(bo.calculate_theta)

##a=bo.simulated_path_Geo
#print("Sigma\n",bo.test())

print(bo.control_variate)
#print(bo.payoff_arithmetic)
#print(bo.calculate_theta)
#print(bo.geometric_standardMC)

#print(bo.payoff_geometric)


bo = BasketOption(S1=100, S2=100, K=100, r=0.05, T=3, σ1=0.3, σ2=0.3, ρ=1,n_steps=15,option_type='put')
print("Put -geo MC vs  closed form:")
#print(bo.calculate_theta)
print(bo.geometric_standardMC)
print(bo.geometric_exact())

print("Put -arith MC vs Control:")
print(bo.arithmetic_standardMC)
#print(bo.payoff_arithmetic)
##print(bo.calculate_theta)

##a=bo.simulated_path_Geo
#print("Sigma\n",bo.test())

print(bo.control_variate)

#print(bo.calculate_theta())

#print("rr\n",bo.test2())

#y = pd.DataFrame(a)
#print(y)
#y.to_csv('A.csv')

bo = BasketOption(S1=5, S2=5, K=5, r=0.15, T=3, σ1=0.7, σ2=0.3, ρ=0.5,n_steps=5,option_type='call')
print("Call - geo MC vs  closed form:")
#print(bo.calculate_theta)
print(bo.geometric_standardMC)
print(bo.geometric_exact())

print("Call -arith MC vs Control:")
print(bo.arithmetic_standardMC)
#print(bo.payoff_arithmetic)
##print(bo.calculate_theta)

##a=bo.simulated_path_Geo
#print("Sigma\n",bo.test())

print(bo.control_variate)
bo = BasketOption(S1=25, S2=75, K=50, r=0.05, T=10, σ1=0.7, σ2=0.3, ρ=0.1,n_steps=10,option_type='put')
print("Put - geo MC vs  closed form:")
#print(bo.calculate_theta)
print(bo.geometric_standardMC)
print(bo.geometric_exact())

print("Put -arith MC vs Control:")
print(bo.arithmetic_standardMC)
#print(bo.payoff_arithmetic)
##print(bo.calculate_theta)

##a=bo.simulated_path_Geo
#print("Sigma\n",bo.test())

print(bo.control_variate)