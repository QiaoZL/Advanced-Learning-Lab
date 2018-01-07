'''
Created on 2017年8月19日

@author: VOIDS
'''

import numpy as np
import scipy as sci

class AlgorithmBase(object):
    '''
     Basic structure of Algorithm class, every algorithm must have same structure
     with the base
    '''
    
    

    def __init__(self, lossFunc, normMethod, optimizeMethod):
        '''
        Constructor
        '''


        