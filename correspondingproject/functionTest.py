
import inputTools

import os
import pandas as pd
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

inputFile = "data/training"
outputFile = "D:/testsuite/output"

def groupAge(age):
    if age < 24:
        return 1.0
    if age >= 24 and age < 34:
        return 2.0
    if age >= 34 and age < 49:
        return 3.0
    if age >= 49:
        return 4.0





userDF = inputTools.sampleInputPd(inputFile)
testuserid = likesDataWithGender = userDF.likesData.groupby('like_id').filter(
    lambda like_id: len(like_id) > 100)

#print(userDF.userData.head(20))

#print(userDF.likesData.head(20))

'''likesDataWithGender = userDF.likesData.groupby('userid').filter(
    lambda userid: len(userid) > 10)'''
likesDataWithGender = likesDataWithGender.groupby('like_id').filter(
    lambda like_id: len(like_id) > 100)
#4930 1871

print(len(likesDataWithGender.groupby('like_id')))
print(len(likesDataWithGender.groupby('userid')))



#print(likesDataWithGender['like_id'].value_counts())

#likesDataWithGender['gender'] = 0
#userGender = pd.DataFrame({"userid": userDF.userData.userid,"gender": userDF.userData.gender})


#userGenderNew = userGender.reindex(range(0,len(likesDataWithGender.index)),method = 'ffill')
likesDataWithGender = pd.merge(userDF.userData,likesDataWithGender,how='right',on='userid')
likesDataWithGender['age_group'] = likesDataWithGender['age'].apply(groupAge)

newLikesFeature = likesDataWithGender.groupby('userid').groups
#print(newLikesFeature)
newLikesFeature = pd.DataFrame.from_dict(newLikesFeature,orient='index')
newLikesFeature['userid'] = 0
newLikesFeature['userid'] = newLikesFeature.index


#print(newLikesFeature)
#newLikesFeature = newLikesFeature.reindex()
#print(newLikesFeature)


print(likesDataWithGender.shape)

#print(likesDataWithGender.head(50))

    #count = 0
    
vectorizer = CountVectorizer()
sumAcc = 0
sumAccA = 0
partSize = round(len(likesDataWithGender.index)/10)
size = len(likesDataWithGender.index)
training = likesDataWithGender.copy()
target = pd.DataFrame(userDF.userData.loc[:,{'gender','age','userid'}])
#print(target)
#training = pd.merge(target,newLikesFeature,how='right',on='userid')
#print(training)
training['age_group'] = training['age'].apply(groupAge)

MNB = MultinomialNB()
LR = LogisticRegression()
CART = tree.DecisionTreeClassifier()
BOOST = ensemble.AdaBoostClassifier()



X = py.array(training.like_id).T
X = X.astype(py.str)
X = vectorizer.fit_transform(X)


print(X)
#A=vectorizer.get_feature_names()
print(vectorizer.vocabulary_)
print(X.shape)
#X = newLikesFeature.ix[:,0:345]

y = py.array(training.gender).T
y = y.astype(py.str)

yage = py.array(training.age_group).T
yage = yage.astype(py.str)

print(cross_val_score(MNB,X,y,cv=10).sum()/10)
print(cross_val_score(MNB,X,yage,cv=10).sum()/10)
#print(cross_val_score(LR,X,y,cv=10).sum()/10)
#print(cross_val_score(LR,X,yage,cv=10).sum()/10)
#print(cross_val_score(BOOST,X,y,cv=10).sum()/10)
#print(cross_val_score(BOOST,X,yage,cv=10).sum()/10)



'''for index in range(1,11):
    
    print(index)
    
    begin = index*partSize
    end = (index+1)*partSize
    
    partA = originDF[0:begin]
    partB = originDF[begin+1:end-1]
    partC = originDF[end:]
    
    training = pd.concat([partA,partC])
    
    X = py.array(training.like_id).T
    X = X.astype(py.str)
    X = vectorizer.transform(X)
    print(X)

    y = py.array(training.gender).T
    y = y.astype(py.str)
    
    yage = py.array(training.age_group).T
    yage = yage.astype(py.str)
    print(y)
    print(yage)
    

    print("Training data done!")   
    clf = MultinomialNB()
    clfAge = MultinomialNB()
    clf.fit(X,y)
    clfAge.fit(X,yage)
    print("Learning done!")
    
    tX = py.array(partB.like_id).T
    tX = tX.astype(py.str)
    tX = vectorizer.transform(tX)
    
    ty = py.array(partB.gender).T
    ty = ty.astype(py.str)
    
    tyage = py.array(partB.age_group).T
    tyage = ty.astype(py.str)
    
    
    sumAcc = sumAcc + metrics.accuracy_score(ty,clf.predict(tX))
    print(sumAcc)
    sumAccA = sumAccA + metrics.accuracy_score(tyage,clfAge.predict(tX))
    print(sumAccA)

print("Gender:",sumAcc/10)
print("Age:",sumAccA/10)'''

   
    















        




