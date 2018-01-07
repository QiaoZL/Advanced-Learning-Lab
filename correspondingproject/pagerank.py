import pandas as pd
import numpy as np


def pagerank (adjmatrix,beta = 0.85):

    rankvector = []
    nnode = max(max(adjmatrix['i']),max(adjmatrix['j']))
    print("number of node:")
    print(nnode)
    
    for i in range(nnode):
        rankvector.append(1/nnode)
        
    rankvector=np.array(rankvector)
    print("original rank vector:")
    print(rankvector)
    
  
    groupsize = adjmatrix.groupby(by='i').size()
    outputnode = adjmatrix.groupby(by='i').groups.keys()
    print("node with output:")
    print(outputnode)
    
    rankMatA = np.zeros([nnode,nnode])
        
    for rowindex in adjmatrix.index:
        rankMatA[adjmatrix.loc[rowindex,'j']-1, adjmatrix.loc[rowindex,'i']-1] = 1/groupsize[adjmatrix.loc[rowindex,'i']]
    
    print("original matrix:")
    print(rankMatA)
        
    #for rowindex in adjmatrix.index:  
        #if adjmatrix.loc[rowindex,'j'] not in outputnode:
            #rankMatA[:,adjmatrix.loc[rowindex,'j']-1] =  1/nnode
    
    
    iternum = 1
    rankMatOutput = np.eye(nnode)
    rankvector = rankvector.reshape((nnode,1))
    
    adjustvector = []
    for i in range(nnode):
        adjustvector.append((1-beta)*(1/nnode))        
    print("(1-beta)*e/n:")
    print(adjustvector)
    adjustvector = np.array(adjustvector)
    adjustvector = adjustvector.reshape((nnode,1))
    
    while 1 :
        rankvectorO = rankvector        
        
        rankvector = beta*np.dot(rankMatA,rankvector) + adjustvector
        rankMatOutput = np.dot(rankMatA,rankMatOutput)
        
        #print(rankvector)
        #print(rankvector == rankvectorO)
        if sum(rankvector == rankvectorO)==nnode:        
            break
        else:
            iternum += 1
            
    print("final matrix:")
    print(rankMatOutput)
    print("final vector:")
    print(rankvector)
    print("iteration times:")
    print(iternum)
    
    
    return 

def datapropoccess(filename = "graph.txt"):
    
    data = open(filename).read()
    data = data.split()
    data = list(map(int,data))
    data = np.array(data)
    data = np.reshape(data,(-1,3))
    
    return data


filename = "graph.txt"
data = datapropoccess(filename)

adjMat = pd.DataFrame(data,columns=['i','j','k'])
print("data:")
print(adjMat)
pagerank(adjmatrix=adjMat)