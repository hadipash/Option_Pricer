# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:46:50 2019

@author: user
"""

import numpy as np

class AmericanOption_BiTree:
    def __init__(self,S,r,sigma,T,step_num,K,t=0):# step_num include node 0
        self.S=S
        self.r=r
        self.K=K
        self.sigma=sigma 
        self.step_num=step_num
        self.dt=(T-t)/(step_num-1)
    
        #up and down definition
        self.up = np.exp(self.sigma * np.sqrt(self.dt)) 
        self.down = np.exp(-self.sigma * np.sqrt(self.dt)) 
        
        #fwd_series_one_step = np.array([np.exp( self.dt * self.r) for i in range(step_num)])# i=0,1,2,3,4,n-1 th node
        
        
        self.up_prob=(np.exp( self.dt * self.r)-self.down)/(self.up -self.down)
        self.down_prob=1-self.up_prob

    def get_stock_tree(self):
        tree=np.ndarray([self.step_num,self.step_num]) # initialize a NxN matrix  
        power_series= np.array(range(self.step_num)) # generate a series for power
        for i in range(self.step_num):
            tree[i,:]= self.S*np.power(self.up,power_series) # assign value line by line
            power_series= power_series-1# next line is the same just power -1 
            
        return tree
    
    def get_call_tree(self):
        C=np.ndarray([self.step_num,self.step_num]) 
        C2=np.ndarray([self.step_num,self.step_num])
        tree=self.get_stock_tree()   #Call self function inside function
        C= np.maximum(tree-self.K, 0.0)# This is American exercise price
        C2=np.copy(C)
        for i in range(self.step_num-1): #calculate backward the option prices
            for j in range(self.step_num-1):
                C2[j][self.step_num-2-j-i]=np.exp(-self.r *self.dt) * ( self.up_prob * C2[j][self.step_num-j-i-1] + self.down_prob * C2[j+1][self.step_num-2-j-i])
                C2[j][self.step_num-2-j-i]= np.maximum(C2[j][self.step_num-2-j-i],C[j][self.step_num-2-j-i])
        return C2
    
    def get_put_tree(self):
        P=np.ndarray([self.step_num,self.step_num]) 
        P2=np.ndarray([self.step_num,self.step_num])
        tree=self.get_stock_tree()   #Call self function inside function
        P= np.maximum(self.K-tree, 0.0)# This is American exercise price
        P2=np.copy(P)
        
        for i in range(self.step_num-1): #calculate backward the option prices
            for j in range(self.step_num-1):
                P2[j][self.step_num-2-j-i]=np.exp(-self.r *self.dt) * ( self.up_prob * P2[j][self.step_num-j-i-1] + self.down_prob * P2[j+1][self.step_num-2-j-i])
                P2[j][self.step_num-2-j-i]= np.maximum(P2[j][self.step_num-2-j-i],P[j][self.step_num-2-j-i])
        return P2

TreeTreeTree=AmericanOption_BiTree(S=50,r=0.05,sigma=0.3,T=0.25,step_num=4,K=50) # Test case 1 :Lecture 7 - P.14 example

test=AmericanOption_BiTree.get_call_tree(TreeTreeTree)
print("Test case 1 :Lecture 7 - P.14 example:\n",test)# up is going right , down is going down

TreeTreeTree=AmericanOption_BiTree(S=50,r=0.05,sigma=0.3,T=0.25,step_num=6,K=50) # Test case 2 :Lecture 7 - P.15 example

test=AmericanOption_BiTree.get_call_tree(TreeTreeTree)
print("Test case 2 :Lecture 7 - P.15 example:\n",test)# up is going right , down is going down


TreeTreeTree=AmericanOption_BiTree(S=50,r=0.05,sigma=0.223144,T=2,step_num=3,K=52) # Test case 3 :Lecture 7 - P.22 example

test=AmericanOption_BiTree.get_put_tree(TreeTreeTree)
print("Test case 3 :Lecture 7 - P.22 example\n,",test)# up is 