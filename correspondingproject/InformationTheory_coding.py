import numpy as np
from itertools import combinations
from scipy.special import perm


class linearcode:
    
    def __init__(self,n=30,k=20):
        self.code = []
        self.n = n
        self.k = k
        self.r = n-k
    
    def generateG(self,seed = 42):
        
        left = np.eye(self.k)       
        
        np.random.seed(seed)
        right = np.random.randint(0,2,size = (self.k,self.r))
        
        self.P = right        
        self.G = np.column_stack((left,right))
        
        return
    
    def generateH(self):
        
        right = np.eye(self.r)
        left = np.transpose(self.P)
        
        self.H = np.column_stack((left,right))
        
        return
    
    def encode(self,m):
        
        return np.matrix((np.dot(m,self.G))%2)
    
    def decode(self,m):
        syn = np.transpose(np.dot(self.H,np.transpose(m))%2)
        
        print("syn",syn)
        print("masseage",m)
        
        synkey = syn.tolist()[0]
        print(any(synkey))
        
        if any(synkey) is False:
            r = m
            return r.tolist()[0][0:self.k]
        else:            
            synkey = tuple(synkey)
            
            evector = self.errorpatterns[synkey]
            print("evector",evector)
            r = (evector + m[0])%2
            print("result",r[0]) 
            return r.tolist()[0][0:self.k]
            
            
    
    def noisy(self,t=3):
        e = np.zeros(shape = (1,self.n))
        e_index = np.random.choice(range(self.n),size = t,replace = False)
        
        for i in e_index:
            e[0,i] = 1        
        
        return e
    
    
    def generate_errortable(self):
        
        errortable = {}
        
        for i in range(1,2**(self.n)):
            evector = self.trans2bit(i,self.n)
            #print(evector) 
            evector = np.matrix(evector)
            
            syn = np.transpose(np.dot(self.H,np.transpose(evector))%2)
            synkey = syn.tolist()[0]
            synkey = tuple(synkey)
            
            if (i == 2**(self.n-4)):
                print(i)
            
            if synkey in errortable.keys():
                if len(errortable[synkey]) > len(evector):
                    errortable[synkey] = evector
            else:
                errortable[synkey] = evector
                    
        self.errorpatterns = errortable
        
        return
    
    
    def generate_errortablefast(self):
        
        errortable = {}
        maxprocessederrors = 0
        processeddict = {}
        
        for i in range(1,2**self.r):
            
            synlist = self.trans2bit(i, self.r)
            synkey = tuple(synlist)
            
            print("processing",synkey)
            
            if synkey not in errortable.keys():                   
                
                for j in range(1,self.n+1):                    
                    
                    if j <= maxprocessederrors:
                        j = maxprocessederrors + 1
                        
                    if j not in processeddict.keys():
                        processeddict[j] = 0
                                                
                    if synkey in errortable.keys():
                        break                    
                                       
                    errorbits = combinations(range(self.n),j)                    
                    numprocessed = 0 
    
                    for errorbit in errorbits:                        
                        
                        error = [0 for i in range(self.n)]
                        for i in errorbit:
                            error[i] = 1                      
                        
                        
                        if numprocessed >= processeddict[j]:                                                    
                            errorv = np.matrix(error)
                            #print(errorv)
                            syni = np.transpose(np.dot(self.H,np.transpose(errorv))%2)
                            synikey = tuple(syni.tolist()[0])                            
                            #print(synikey)                    
                            
                            
                            if synikey == synkey:
                                errortable[synikey] = errorv
                                processeddict[j] = numprocessed                            
                                break  
                            
                            if synikey not in errortable.keys():
                                errortable[synikey] = errorv
                        
                        numprocessed += 1    
                    
                    maxprocessederrors = j           
                        
                          
                        

        self.errorpatterns = errortable
        return
    
    def generate_errortablebest(self):
        
        errortable = {}        
        
        for numerrors in range(1,self.n+1):            
            if(len(errortable.keys()) >= 2**self.r):
                break
            
            errorbits = combinations(range(0,self.n),numerrors)
            
            for errorbit in errorbits:
                if(len(errortable.keys()) >= 2**self.r):
                    break
                
                error = [0 for i in range(self.n)]
                for i in errorbit:
                    error[i] = 1  
                
                errorv = np.matrix(error)
                syni = np.transpose(np.dot(self.H,np.transpose(errorv))%2)
                synikey = tuple(syni.tolist()[0])
                
                if synikey not in errortable.keys():
                    errortable[synikey] = errorv
            
        self.errorpatterns = errortable
        return
    
    def trans2bit(self,k,n= 10):
        
        r = bin(k)[2:]
        
        if len(r) < n:
            for j in range(n-len(r)):
                r = '0' + r
        
        return [int(j) for j in r]
    
    @classmethod
    def hammingdistance(cls,a,b):
        
        return np.sum((a+b)%2)



n = 30
k = 20
t = 1

testcode = linearcode(n,k)
testcode.generateG(seed = 42)
print(testcode.G.shape)
testcode.generateH()
print(testcode.H.shape)
  
m = np.random.randint(0,10,size = (1,k))%2
  
c = testcode.encode(m)
e = testcode.noisy(t)
  
x = (c + e)%2
  
testcode.generate_errortablebest()
print(len(testcode.errorpatterns.keys()))
r = testcode.decode(x)
  
print("e",e.tolist()[0])
print ("m",m.tolist()[0])
print ("x",x.tolist()[0])
print ("r",[int(i) for i in r])
  
mind = n
for i in range(k):
    for j in range(i+1,k):
        d = testcode.hammingdistance(testcode.G[i,:], testcode.G[j,:])
        if testcode.G[i,:].tolist() == testcode.G[j,:].tolist():
            print(i,j)
        if d < mind:
            mind = d
  
print("least hamming distance",mind)

# G = [[1, 0, 0, 1, 0, 1],[0,1,0,0,1,1],[0,0,1,1,1,0]]
# H = [[1,0,1,1,0,0],[0,1,1,0,1,0],[1,1,0,0,0,1]]
# G = np.matrix(G)
# M = np.matrix([1,0,1])
# R = np.matrix([1,1,1,0,0,1])
# print((np.dot(M,G))%2)
# print((np.dot(H,np.transpose(R)))%2)


