
import inputTools 

import os
import pandas as pd
import numpy as py

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree,ensemble, feature_selection,preprocessing,metrics, linear_model

from learningTools import likesIDGenderMNB
from sklearn.feature_extraction.text import HashingVectorizer
from bokeh.core.properties import String


import learningTools
import inputTools
import sklearn.feature_selection
import sklearn.linear_model

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve
from sklearn.svm.classes import LinearSVR, LinearSVC
from sklearn.utils import class_weight

inputFile = "data/training"
outputFile = "D:/testsuite/output"
testFile = "D:/testsuite/public-test-data"

def groupAge(age):
    if age < 24:
        return 1.0
    if age >= 24 and age < 34:
        return 2.0
    if age >= 34 and age < 49:
        return 3.0
    if age >= 49:
        return 4.0

class clfSelector():
    
    coef_ = 0
    



userDF = inputTools.sampleInputPd(inputFile)
testDF = inputTools.sampleInputPd(testFile)

userDF.featureData.rename(columns={'userId':'userid'},inplace=True)

#print(userDF.userData.head(20))

#print(userDF.likesData.head(20))

   


#print(likesDataWithGender)

#likesDataWithGender['gender'] = 0
userGender = userDF.userData.copy()
userGender['age_group']=userGender['age'].apply(groupAge)

LIWC = userDF.featureData
userAge = userDF.userData.copy()
userAge['age_group']=userAge['age'].apply(groupAge)
userAge = pd.merge(LIWC,userAge,on='userid',how= 'right')

#print(userAge.groupby('age_group').size())
print(userAge.groupby('gender').size())
print(userAge.groupby('age_group').size())


#selector = feature_selection.SelectFromModel

X=userAge.ix[:,'WC':'AllPct']



feature = ['ope','con','ext','agr','neu']
featureSize = [40,40,40,40,40]
cost = [1,1,1,1,1]
ga = [0.0001,0.0001,0.00001,0.0001,'auto']

testindex = 4
testfeature = feature[testindex]
testsize = featureSize[testindex]
testcost = cost[testindex]
testga = ga[testindex]

#exo may need select


ya=userAge.ix[:,'age_group']
yf=userAge.ix[:,feature[testindex]]
yg=userAge.ix[:,'gender']

selector = feature_selection.SelectKBest(score_func=feature_selection.f_regression
                                         ,k=testsize)




scalerX = preprocessing.Normalizer(norm='l2')
standardScalerX = preprocessing.StandardScaler()
#scalerYa = preprocessing.Normalizer(norm='l2')
#scalerYg = preprocessing.Normalizer(norm='l2')
#scalerYf = preprocessing.Normalizer(norm='l2')

scalerX.fit(X)
X=scalerX.transform(X)
#X=standardScalerX.fit_transform(X)
print(X.shape)

lars_cv = linear_model.LassoLarsCV(cv=6).fit(X, yf)
alphas = py.linspace(lars_cv.alphas_[0], .1 * lars_cv.alphas_[0], 6)
treeScore = linear_model.RandomizedLasso(alpha=alphas,random_state=42)
#treeScore.fit(X, yf)


#X=selector.fit_transform(X,yf)

#trees = ensemble.ExtraTreesRegressor(100).fit(X, yf)
print(lars_cv.coef_)
#lars_cv.coef_ = py.abs(lars_cv.coef_)

treeSelector = feature_selection.SelectFromModel(lars_cv,prefit=True)

#treeSelector = feature_selection.SelectFromModel(treeScoreSaver,prefit=True,threshold=0.5)

print(treeSelector.get_params())

#X=treeSelector.transform(X)

X2=treeSelector.transform(X)
X=treeScore.fit_transform(X,yf)

#print(lars_cv.coef_)\
print(X2.shape)
print(X.shape)
#print(yf.shape)  

sumAcc = 0
count = 0

MNBfeature = MultinomialNB()
linearsvrfeature = LinearSVR(loss='squared_epsilon_insensitive',C=testcost)
linearsvcfeature = LinearSVC(loss='squared_epsilon_insensitive',C=testcost)
lasso = linear_model.Lasso(alpha=alphas[0])
print(alphas)
#linearsvrfeature.fit(X, yf)

#print(cross_val_score(MNBfeature,X,ya,cv=10).sum()/10)
#print(cross_val_score(linearsvcfeature,X,ya,cv=10).sum()/10)
#print(cross_val_score(MNBfeature,X,yg,cv=10).sum()/10)
#print(cross_val_score(linearsvcfeature,X,yg,cv=10).sum()/10)
#print(cross_val_score(svcGender,X,yg,cv=5).sum()/5)

print(testfeature)
print('SVM')
print('larsCV')
print(X2.shape)
result = cross_val_score(linearsvrfeature,X2,yf,cv=10,scoring = 'mean_squared_error')
print(result)
for score in result:
    count += py.sqrt(-score)
print(count/10)
count=0
print('LASSO')
print(X.shape)
result = cross_val_score(linearsvrfeature,X,yf,cv=10,scoring = 'mean_squared_error')
print(result)
for score in result:
    count += py.sqrt(-score)
print(count/10)

'''print('LASSO')
count=0
result = cross_val_score(lasso,X2,yf,cv=10,scoring = 'mean_squared_error')
print(result)
for score in result:
    count += py.sqrt(-score)
print(count/10)
count=0
result = cross_val_score(lasso,X,yf,cv=10,scoring = 'mean_squared_error')
print(result)
for score in result:
    count += py.sqrt(-score)
print(count/10)

#print(cross_val_score(svcGender,X,yg,cv=10).sum()/10)'''

#print(userGender.loc[:,'gender'])
#print(userDF.userData.loc[:,'gender'])
#print(metrics.accuracy_score(userGender.loc[:,'gender'],userDF.userData.loc[:,'gender']))
#print(metrics.accuracy_score(userGender.loc[:,'age_group'],userDF.userData.loc[:,'age_group']))



   
    















        





