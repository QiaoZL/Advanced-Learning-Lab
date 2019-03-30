
import inputTools

import os

import numpy as py
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn import tree
from sklearn import metrics
from dask.dataframe.core import DataFrame
from learningTools import likesIDGenderMNB
from sklearn.feature_extraction.text import HashingVectorizer,CountVectorizer
from bokeh.core.properties import String

from sklearn.cross_validation import cross_val_score
from unittest.mock import inplace
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import ensemble
from sklearn.svm.classes import LinearSVC
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn import discriminant_analysis
import os

def groupAge(age):
    if age < 24:
        return 1.0
    if age >= 24 and age < 34:
        return 2.0
    if age >= 34 and age < 49:
        return 3.0
    if age >= 49:
        return 4.0



inputFile = "data/training"
outputFile = "D:/testsuite/output"
testFile = "D:/testsuite/public-test-data"

userDF = inputTools.sampleInputPd(inputFile)
testDF = inputTools.sampleInputPd(testFile)
'''likesdata = userDF.likesData.groupby('like_id').filter(
        lambda like_id: len(like_id) > 2)
print(len(likesdata.groupby('like_id').groups))'''

likesize = 50
likesdata = inputTools.likesDataprepocessing(userDF,likesize)

print(likesdata.columns[0:-3])
print(likesdata.shape)

MNB = MultinomialNB()#0.71,0.48
#GNB = GaussianNB() #Too bad
LR = LogisticRegression()#0.74,0.544
CART = tree.DecisionTreeClassifier()#0.658 0.49
BOOST = ensemble.AdaBoostClassifier()#0.699 0.52
SVC = LinearSVC() #0.72,0.51
#QDA = discriminant_analysis.QuadraticDiscriminantAnalysis()
#lasso = linear_model.Lasso()
#GPC = gaussian_process.GaussianProcessClassifier()

X=likesdata.ix[:,likesdata.columns[0:-3]]
print(X.shape)
yg=likesdata['gender']
ya=likesdata['age_group']

print('MNB')
print(cross_val_score(MNB,X,yg,cv=10).sum()/10)
print(cross_val_score(MNB,X,ya,cv=10).sum()/10)
print('LR')
print(cross_val_score(LR,X,yg,cv=10).sum()/10)
print(cross_val_score(LR,X,ya,cv=10).sum()/10)
print('CART')
print(cross_val_score(CART,X,yg,cv=10).sum()/10)
print(cross_val_score(CART,X,ya,cv=10).sum()/10)
print('BOOST')
print(cross_val_score(BOOST,X,yg,cv=10).sum()/10)
print(cross_val_score(BOOST,X,ya,cv=10).sum()/10)
print('SVC')
print(cross_val_score(SVC,X,yg,cv=10).sum()/10)
print(cross_val_score(SVC,X,ya,cv=10).sum()/10)
'''print('QDA')
print(cross_val_score(QDA,X,yg,cv=10).sum()/10)
print(cross_val_score(QDA,X,ya,cv=10).sum()/10)'''

#print(userDF.userData.userid)
'''userDF.featureData.rename(columns={'userId':'userid'},inplace=True)
personalityData = pd.merge(userDF.featureData,userDF.userData,on='userid',how= 'right')

likesData = pd.merge(userDF.userData,userDF.likesData,on='userid',how= 'right')
likesData = likesData.groupby('like_id').filter(
    lambda like_id: len(like_id) > 100)
likesData = likesData.reindex()
print(likesData.index)
userlikegroup = likesData.groupby('like_id').groups
useridgroup = likesData.groupby('userid').groups

#print(userlikegroup.groups)

newfeature = pd.DataFrame(columns=userlikegroup.keys())
newfeature['userid']=userDF.userData['userid']
newfeature['gender']=userDF.userData['gender']
newfeature['age_group']=userDF.userData['age'].apply(groupAge)

newfeature = newfeature.fillna(0)
#newfeature = map()
print(newfeature)

#tempindex = likesData.loc[0,'userid']
#print(likesData.loc[0,['userid']])
for key in useridgroup.keys():
    print(key)
    iuserid = key
    tempindex= useridgroup[iuserid]
    target = likesData.ix[tempindex,['like_id']]#.reindex()
    #print(target)
    newfeature.ix[newfeature['userid'] == iuserid,target['like_id']]=1
    #print(newfeature.ix[newfeature['userid'] == iuserid,target['like_id']]) #replace(0,1,inplace=True)
    
print(newfeature)
    
#print(userlikegroup.indices)
      
#index,groupA = likesData.groupby('userid')
#column,groupB = likesData.groupby('likeid')

#print(groupA)

#print(index)
#print(column)'''



'''a = len('1251e9c40aa02337ae05f252a3af8066')
b = len('D:/testsuite/public-test-data\\text\\')
c = len('D:/testsuite/training\\text\\')

i=0
for name in userDF.textData.filenames:
    
    userDF.textData.filenames[i] = name[-4-a:-4]
    i+=1

print(userDF.textData.filenames)
print(len(userDF.textData.filenames))

tfidfVectorizer = TfidfVectorizer()


print('GNB')
print(cross_val_score(GNB,X,yg,cv=10).sum()/10)
print(cross_val_score(GNB,X,ya,cv=10).sum()/10)
'''
