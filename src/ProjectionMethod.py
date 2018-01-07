'''
Created on 2017年8月29日

@author: VOIDS
'''
from sklearn.metrics.pairwise import rbf_kernel
class RBFKernel():
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    def calculate(self,X,Y,selfgamma = 1):
        result = rbf_kernel(X,Y,gamma = selfgamma)        
        return result
    
    def phi(self,X):
        phix = 0
        
        return phix
        