from sklearn.metrics.pairwise import rbf_kernel
from numpy.linalg import norm
import numpy as np
from src import LinearCombination


process = LinearCombination.kernelGramSchmidtProcess(rbf_kernel)
np.random.seed(42)
data = np.random.random(size=[10,5])
process.fit(data)
print(process.kmatrix)
print(process.basisweight)
