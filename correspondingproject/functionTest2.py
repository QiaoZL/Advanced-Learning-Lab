
import inputTools

import os
import pandas as pd
import numpy as py
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import decomposition
from sklearn.svm import SVR
import matplotlib.pyplot as plt 

from sklearn import tree
from sklearn import metrics
from dask.dataframe.core import DataFrame
from learningTools import likesIDGenderMNB
from sklearn.feature_extraction.text import HashingVectorizer
from bokeh.core.properties import String

inputFile = "D:/testsuite/training"
outputFile = "D:/testsuite/output"

userDF = inputTools.sampleInputPd(inputFile)

#print({userDF.userData['userid'],userDF.userData.loc[:,userDF.userData.columns[4:]]})
userDF.featureData.rename(columns={'userId':'userid'},inplace=True)
LIWC = userDF.featureData
personalityData = pd.merge(LIWC,userDF.userData,on='userid',how= 'right')
#print(range(personalityData.columns.index('WC'),personalityData.columns.index('OtherP')))
features = ['ope','con','ext','arg','neu']

print("Training data Done!")

#pca = decomposition.PCA(n_components='mle')
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

X=personalityData.ix[:,'WC':'OtherP']
y=personalityData.ix[:,features[0]]
svr_rbf.fit(X,y)
result = svr_rbf.predict(X)
print(result)
print("Learning Done!")

score = y.sum()-result.sum()

print(score)
    
#print(svr_rbf.predict(X))
#print(metrics.classification_report(y,svr_rbf.predict(X)))

#pca.fit(personalityData.ix[:,'WC':'OtherP'],personalityData.ix[:,features[0]])

#print(pca.fit_transform(personalityData.ix[:,'WC':'OtherP'],personalityData.ix[:,features[0]]))
#print(pca.n_components_)
print(features[0])

#print(personalityData.ix[:,'WC':'OtherP'])
#LIWCTrainingData = merge 



