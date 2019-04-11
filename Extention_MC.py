#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 00:03:06 2019

@author: Esuberante
Asian Option evaluation using standard Monte Carlo
"""
import numpy as np
from black_scholes import BlackScholes

np.random.seed(51)

class ExtentionMonteCarlo:

    def __init__(self, S, K, r, T, σ, n, option_type, m=int(1e5), t=0.0):
        self.__S, self.__K, self.__r = S, K, r
        self.__σ, self.__Δ = σ, T - t
        self.__n, self.__m, self.__dt = n, m, T / n
        self.__option_type = option_type.lower()

    def european_payoff(self):
        s_path = np.ndarray(shape=(self.__n+1, self.__m), dtype=float)
        s_path[0] = self.__S
        drift = np.exp((self.__r - (self.__σ ** 2) / 2) * self.__dt)
        for step in range (1, self.__n+1):
            growth_factor = drift * np.exp(self.__σ * np.sqrt(self.__dt) * np.random.standard_normal(self.__m))
            s_path[step] = s_path[step-1] * growth_factor

        if self.__option_type == 'call':
            return np.exp(-self.__r * self.__Δ) * np.mean(np.maximum(s_path[-1] - self.__K, 0))
        elif self.__option_type == 'put':
            return np.exp(-self.__r * self.__Δ) * np.mean(np.maximum(self.__K - s_path[-1], 0))

    def american_payoff(self):
        s_path = self.__S *np.exp(np.cumsum((self.__r - (self.__σ ** 2) / 2) * self.__dt + self.__σ * np.sqrt(self.__dt) * np.random.standard_normal((self.__n + 1, self.__m)), axis=0))
        s_path[0] = self.__S
        h = np.ndarray(shape=(self.__n+1, self.__m), dtype=float)
        
        if self.__option_type == 'call':
            h = np.maximum(s_path - self.__K, 0)
        elif self.__option_type == 'put':
            h = np.maximum(self.__K - s_path, 0)
                
        v = h[-1]
        #calculation by backward induction
        for step in range(self.__n - 1, 0, -1):
            rg = np.polyfit(s_path[step], v * np.exp(-self.__r * self.__dt), 5)
            cv = np.polyval(rg, s_path[step]) #continuation values
            v = np.where(h[step] > cv, h[step], v * np.exp(-self.__r * self.__dt))
        return np.exp(-self.__r * self.__dt )* np.mean(v)
    
    def barrier_upandout(self, barrier):
        sum=0
        bsm = BlackScholes(self.__S, self.__K, self.__r, self.__Δ, self.__σ)
        for j in range(0, self.__m):
            sT=self.__S
            out=False
   
            for i in range(0,int(self.__n)):
                z=np.random.normal()
                sT*=np.exp((self.__r - 0.5 * self.__σ *self.__σ ) * self.__dt +  self.__σ * z * np.sqrt(self.__dt))
                if sT>barrier:
                    out=True
       
            if self.__option_type == 'call':
                if out==False:
                    sum+=bsm.call()
            if self.__option_type == 'put':
                if out==False:
                    sum+=bsm.put()
                    
        return sum/self.__m

    def barrier_downandout(self, barrier):
        sum=0
        bsm = BlackScholes(self.__S, self.__K, self.__r, self.__Δ, self.__σ)
        for j in range(0, self.__m):
            sT=self.__S
            out=False
   
            for i in range(0,int(self.__n)):
                z=np.random.normal()
                sT*=np.exp((self.__r - 0.5 * self.__σ *self.__σ ) * self.__dt +  self.__σ * z * np.sqrt(self.__dt))
                if sT< barrier:
                    out=True
       
            if self.__option_type == 'call':
                if out==False:
                    sum+=bsm.call()
            if self.__option_type == 'put':
                if out==False:
                    sum+=bsm.put()
                    
        return sum/self.__m
    
    def barrier_upandin(self, barrier):
        sum=0
        bsm = BlackScholes(self.__S, self.__K, self.__r, self.__Δ, self.__σ)
        for j in range(0, self.__m):
            sT=self.__S
            inop=False
   
            for i in range(0,int(self.__n)):
                z=np.random.normal()
                sT*=np.exp((self.__r - 0.5 * self.__σ *self.__σ ) * self.__dt +  self.__σ * z * np.sqrt(self.__dt))
                if sT> barrier:
                    inop=True
       
            if self.__option_type == 'call':
                if inop==True:
                    sum+=bsm.call()
            if self.__option_type == 'put':
                if inop==True:
                    sum+=bsm.put()
                    
        return sum/self.__m
    
    def barrier_downandin(self, barrier):
        sum=0
        bsm = BlackScholes(self.__S, self.__K, self.__r, self.__Δ, self.__σ)
        for j in range(0, self.__m):
            sT=self.__S
            inop=False
   
            for i in range(0,int(self.__n)):
                z=np.random.normal()
                sT*=np.exp((self.__r - 0.5 * self.__σ *self.__σ ) * self.__dt +  self.__σ * z * np.sqrt(self.__dt))
                if sT<barrier:
                    inop=True
       
            if self.__option_type == 'call':
                if inop==True:
                    sum+=bsm.call()
            if self.__option_type == 'put':
                if inop==True:
                    sum+=bsm.put()
                    
        return sum/self.__m
        
    
#test1 = ExtentionMonteCarlo(40, 40, 0.05, 0.5, 0.2, 100, 'put')
#print(test1.barrier_upandout(42))
#print(test1.barrier_upandin(42))


