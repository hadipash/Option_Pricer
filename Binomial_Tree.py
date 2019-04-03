# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:46:50 2019

@author: user
"""

import numpy as np

class BiTree:
    def __init__(self,S,r,sigma,t,T,step_num):# step_num include node 0
        self.S=S
        self.r=r
        self.sigma=sigma 
        self.step_num=step_num
        self.dt=(T-t)/(step_num-1)
    
        #up and down definition
        self.up = np.exp(self.sigma * np.sqrt(self.dt)) 
        self.down = np.exp(-self.sigma * np.sqrt(self.dt)) 
        
        fwd_series_one_step = np.array([np.exp( self.dt * self.r) for i in range(step_num)])# i=0,1,2,3,4,n-1 th node
        
        
        self.up_prob=(fwd_series_one_step-self.down)/(self.up -self.down)
        self.down=1-self.up_prob

    def get_tree(self):
        tree=np.ndarray([self.step_num,self.step_num]) # initialize a NxN matrix  
        power_series= np.array(range(self.step_num)) # generate a series for power
        for i in range(self.step_num):
            tree[i,:]= self.S*np.power(self.up,power_series) # assign value line by line
            power_series= power_series-1# next line is the same just power -1
        return tree

fivestepTree=BiTree(5,0.05,0.1,0,1,5)

test=BiTree.get_tree(fivestepTree)
print(test)
