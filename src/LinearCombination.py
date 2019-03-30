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
        result += weightxi*kernelFunc(data[xi],y)
        xi += 1   
        
    return result

class kvector ():
    
    def __init__ (self):
        self.weight = np.array()
        self.data = []   

class kernel_GramSchmidtProcess():    
       
    def __init__(self,kernelFunc):
        self.kernelFunc = kernelFunc
        self.basis = []
        self.basisweight = []
        self.kmatrix = []
    
    def fit(self,data,gamma = 0):
        self.basisdata = data
        self.generateKmatrix(data,gamma)
        self.basisweight = self.generateBasis(data)
        
        return self.basisweight

    def transform(self,origindata,y=None):
        m = origindata.shape[0]
        n = len(self.basisweight[0])
        transKMatrix = np.array(self.kernelFunc(origindata,self.data))
        print(transKMatrix.shape)
        
        result = np.array(np.eye(m,n))
        
        for i in range(m):
            
            for j in range(n):
                
                result[i,j] = np.sum(transKMatrix[i] * self.basisweight[j])
        
        self.transdata = result
        
        return result
        
    def generateKmatrix(self,data,gamma = 0):
        
        self.data = data
        print("Generate Kernel Matrix")
        if gamma == 0 :
            self.kmatrix = np.array(self.kernelFunc(data))
        else:
            self.kmatrix = np.array(self.kernelFunc(data,gamma))
            
        print("Done")
        return        
        
    def generateBasis(self,X):
        m = X.shape[0]        
        
        basisweight = np.eye(m,m)
        phi = np.eye(m,m) 
        for i in range(m):
            for j in range(i):
                basisweight[i] -= self.k_inner_product(phi[i], basisweight[j])*basisweight[j]
            basisweight[i] /= self.k_norm(basisweight[i])
            
        return basisweight
        
    def k_inner_product(self,weightx,weighty):
        
        result = np.array(np.multiply(np.matrix(weightx),np.transpose(np.matrix(weighty)))) * np.array(self.kmatrix)
        
#         for weightxi in weightx:
#             yi = 0
#             for weightyi in weighty:
#                 result += km[xi,yi]*weightxi*weightyi
#                 yi += 1            
#             xi +=1
                        
        return np.sum(result)
    
    def k_norm(self,weight,normtype = 'L2'):
        if normtype is 'L2':
            result = self.k_inner_product(weight,weight)
            result = math.sqrt(abs(result))
        return result
    


               


            
            
            
            
            
            
            
            
            
                