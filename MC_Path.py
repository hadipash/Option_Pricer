# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np


class MC_Path:

    def __init__(self, S, K, r, T, σ, n, m=int(1e4), t=0.0, ControlVariate=False):
        self.__S, self.__K, self.__r = S, K, r
        self.__σ, self.__Δ = σ, T - t
        self.__n, self.__m, self.__dt = n, m, T / n
        self.__S_path = []
        

    def __gen_paths(self):
        
        drift = np.exp((self.__r - (self.__σ ** 2) / 2) * self.__dt)#1step :exp(rdt)
        gf = drift * np.exp(self.__σ * np.sqrt(self.__dt) * np.random.normal(size=(self.__m, self.__n )))# M(samples) x n(time steps) , exp(rdt)

        for i in range(self.__m):
            self.__S_path.append([self.__S * gf[i][0]]) # time step =1  OR  i = 0 for all M paths : S(1) = S(0) * exp(rdt - sigma^2 *dt /2) * exp(sigma *sqrt(dt) * dZ)
            for j in range(1, self.__n): #N-1 time steps here
                self.__S_path[i].append(self.__S_path[i][j - 1] * gf[i][j]) #generate mote calo for M sample 

    def get_paths(self):
        self.__gen_paths() #run the generation first
        return np.array(self.__S_path) #Get the array results
  
ao = MC_Path(S=100, K=100, r=0.05, T=5, σ=0.001, n=5,m=7)
print("MC path test for 7 paths with 5 time steps",ao.get_paths())



