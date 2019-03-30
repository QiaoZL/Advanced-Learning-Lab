from sklearn.metrics.pairwise import rbf_kernel

import numpy as np
from src import LinearCombination
from sklearn import datasets

import sklearn_relief as relief
from skfeature.function.similarity_based import lap_score
from sklearn import feature_selection


process = LinearCombination.kernel_GramSchmidtProcess(rbf_kernel)
#np.random.seed(42)
#data = np.random.random(size=[10,5])
#process.fit(data)
#print(process.kmatrix)
#print(process.basisweight)
Xg,yg = datasets.make_gaussian_quantiles(n_features=10,random_state=42)
print(yg.shape)
Xp,yp = datasets.make_multilabel_classification(n_features=10,random_state=42,n_classes= 1)
print(yp.shape)

Rf1 = relief.Relief()
print(Rf1.fit(Xg,yg).w_)
Rf2 = relief.ReliefF()
print(Rf2.fit(Xg,yg).w_)
Rf3 = relief.RReliefF()
print(Rf3.fit(Xg,yg).w_)
L_score = lap_score.lap_score(Xg)
print(L_score)
MI = feature_selection.mutual_info_classif(Xg,yg)
print(MI)

数学 = "aaa"
print(数学)

# a = 154476802108746166441951315019919837485664325669565431700026634898253202035277999
# b = 36875131794129999827197811565225474825492979968971970996283137471637224634055579
# c = 4373612677928697257861252602371390152816537558161613618621437993378423467772036
# print( (a/(b+c)) + (b/(a+c)) + (c/(a+b)) )