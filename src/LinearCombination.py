'''
Created on 2017年12月8日

@author: VOIDS
'''
import numpy as np
import math


def getValueOnBasis(data, weightx, y, kernelFunc, gamma = 0):
    
    result = 0
    xi = 0    
   
    for weightxi in weightx:
        result += weightxi*kernelFunc(result[xi],y)
        xi += 1   
        
    return result

class kvector ():
    
    def __init__ (self):
        self.weight = np.array()
        self.data = []   

class kernelGramSchmidtProcess():    
       
    def __init__(self,kernelFunc):
        self.kernelFunc = kernelFunc
        self.basis = []
        self.basisweight = []
        self.kmatrix = []
    
    def fit(self,data,gamma = 0):
        self.basisdata = data
        self.generateKmatrix(data,gamma)
        self.basisweight = self.generateBasis(data)

    def transform(self,data):
        return
        
    def generateKmatrix(self,data,gamma = 0):
        
        self.data = data
        if gamma == 0 :
            self.kmatrix = self.kernelFunc(data,data)
        else:
            self.kmatrix = self.kernelFunc(data,data,gamma)
        return        
        
    def generateBasis(self,X):
        m = X.shape[0]        
        
        basisweight = np.eye(m,m)
        vectormu = np.eye(m,m) 
        for i in range(m):            
            if i != 0:
                for j in range(i):
                    basisweight[i] -= self.k_inner_product(vectormu[i], basisweight[j])*basisweight[j]
                basisweight[i] /= self.k_norm(basisweight[i])
            
        return basisweight
        
    def k_inner_product(self,weightx,weighty):
    
        result = 0
        xi = 0       
                
        for weightxi in weightx:
            yi = 0
            for weightyi in weighty:
                result += self.kmatrix[xi,yi]*weightxi*weightyi
                yi += 1            
            xi +=1
                        
        return result
    
    def k_norm(self,weight,normtype = 'L2'):
        if normtype is 'L2':
            result = self.k_inner_product(weight,weight)
            result = math.sqrt(result)
        return result
            
            
               


            
            
            
            
            
            
            
            
            
                