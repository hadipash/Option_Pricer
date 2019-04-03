# -*- coding: utf-8 -*-

import numpy  as np
from numpy.random import mtrand

 # this is to fullfill requirement of  "reproducible" , to generate a set of random number and call it based on seed and path_num
Path_random_number='C:/Users/user/Downloads/random_number.npy'



class MC_Random_Number_generator:
    def __init__(self,seed,path_num,CorrelationMatrix=np.array([1])):
        self.CorrelationMatrix = CorrelationMatrix
        #pct_to_target = np.random.normal(0, 1, 9)
        self._seed=seed
        self.path_num=path_num
        self.randon_generator=np.random.mtrand.RandomState(seed=self._seed)
        self.random_number=np.load(Path_random_number)
        
    def get_normal_random(self):
        Number_Underlying = self.CorrelationMatrix.shape[0]
        
        if Number_Underlying==1:
            result = self.random_number[0][:self.path_num] # get the first N random number
            self.random_number=self.random_number[1:] # shuffle for next time
            return result
        else:
            result = self.randon_generator.randn(Number_Underlying,self.path_num)
            """
            Return the Cholesky decomposition, no so sure , to be checked ,seems not correct
            """
            cholesky_result = np.linalg.cholesky(self.CorrelationMatrix)
            return np.dot(cholesky_result,result)
        

s=MC_Random_Number_generator(5,100,np.array([[1.,0.9],[0.9,1.]]))

a=MC_Random_Number_generator.get_normal_random(s)

s2=MC_Random_Number_generator(1,100)

a2=MC_Random_Number_generator.get_normal_random(s2)

print("with correlation matrix, size:" , a.shape)
print("without correlation matrix, size:" , a2.shape)