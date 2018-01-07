
import pandas as pd
import numpy as py

from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.svm import SVC


def groupAge(age):
    if age < 24:
        return 1.0
    if age >= 24 and age < 34:
        return 2.0
    if age >= 34 and age < 49:
        return 3.0
    if age >= 49:
        return 4.0

def simpleLearning(trainingData):
    
    datasize = len(trainingData)
    
    sumAgeType1 = 0
    sumAgeType2 = 0
    sumAgeType3 = 0
    sumAgeType4 = 0
    
    sumAge = 0
    
    sumGender1 = 0
    sumGender2 = 0
    
    sumOpe = 0
    sumCon = 0
    sumExt = 0
    sumArg = 0
    sumNeu = 0
    
    for user in trainingData:
        
        
        sumAge = sumAge + user.age
        
        if user.age <= 24:
            sumAgeType1 = sumAgeType1+1
        if user.age > 24 and user.age < 34:
            sumAgeType2 = sumAgeType2+1
        if user.age > 34 and user.age <= 49:
            sumAgeType3 = sumAgeType3+1
        if user.age > 49:
            sumAgeType4 = sumAgeType4+1
            
        if user.gender == 0:
            sumGender1 = sumGender1 + 1
        else:
            sumGender2 = sumGender2 + 1
            
        sumOpe = sumOpe + user.open
        sumCon = sumCon + user.conscientious
        sumExt = sumExt + user.extrovert
        sumArg = sumArg + user.agreeable
        sumNeu = sumNeu + user.neurotic
        
        print(sumAge)
    
    averageClassifier = []
    averageClassifier.append(sumAge/datasize)
    averageClassifier.append(sumGender2/datasize)
    averageClassifier.append(round(sumOpe/datasize,2))
    averageClassifier.append(round(sumCon/datasize,2))
    averageClassifier.append(round(sumExt/datasize,2))
    averageClassifier.append(round(sumArg/datasize,2))
    averageClassifier.append(round(sumNeu/datasize,2))
    
    return averageClassifier



def likesIDGenderMNB(userDF,vectorizer):
        
    likesDataWithGender = userDF.likesData.copy()
    
    userGender = pd.DataFrame({"userid": userDF.userData.userid,"gender": userDF.userData.gender})
    
    likesData = userDF.likesData.groupby('userid').filter(
        lambda userid: len(userid) > 10)   
    likesData = likesData.groupby('like_id').filter(
        lambda like_id: len(like_id) > 100)
    
    likesDataWithGender = pd.merge(userGender,likesData,how='right',on='userid')
    
    
    #vectorizer = HashingVectorizer(non_negative=True)
    
    X = py.array(likesDataWithGender.like_id).T
    X = X.astype(py.str)    
    X = vectorizer.fit_transform(X)

    y = py.array(likesDataWithGender.gender).T
    y = y.astype(py.str)


    print("Training data Done!")   

    clf = MultinomialNB()
    clf.fit(X,y)
    
    print("Learning Done!")
    
    return clf

def predictGenderLikesid(userDF,clf,vectorizer):
    
    userGender = pd.DataFrame({"userid": userDF.userData.userid,"gender": userDF.userData.gender})
    likesDataWithGender = userDF.likesData
    
    #vectorizer = HashingVectorizer(non_negative=True)
    userGender = pd.merge(userGender,likesDataWithGender,how='right',on='userid')
    
    X = py.array(userGender.like_id).T
    X = X.astype(py.str)
    X = vectorizer.transform(X)
    
    prediction = clf.predict(X)
    print(prediction)
    
    userGender['gender'] = prediction.astype(py.float)
    
    
    
    for l,group in userGender.groupby('userid'):
        #print(userDF.userData.loc[userDF.userData['userid']==l,'gender'])
        
        if group['gender'].sum() >= len(group.index)/2:
            
            userDF.userData.loc[userDF.userData['userid']==l,'gender'] = 1
            
        else:
            
            userDF.userData.loc[userDF.userData['userid']==l,'gender'] = 0
    
    
    return
        
           

def likesIDAgeMNB(userDF,vectorizer):
    
    clfAge = MultinomialNB()
    userAge = userDF.userData
    userAge['age_group'] = userDF.userData['age'].apply(groupAge)
    
    likesData = userDF.likesData.groupby('userid').filter(
        lambda userid: len(userid) > 10)   
    likesData = likesData.groupby('like_id').filter(
        lambda like_id: len(like_id) > 100)
    
    userAge = pd.merge(userAge,likesData,how='right',on='userid')
  
    print("Training data Done!")
    #vectorizer = HashingVectorizer(non_negative=True)
    
    X = py.array(userAge.like_id).T
    X = X.astype(py.str)
    X = vectorizer.fit_transform(X)
    
    yage = py.array(userAge.age_group).T
    yage = yage.astype(py.str) 
    clfAge.fit(X,yage)
    
    print("Learning Done!")
    
    return clfAge

def predictAgeLikesid(userDF,clf,vectorizer):
    
    userAge = userDF.userData
    userDF.userData['age_group'] = 0
    userAge['age_group'] = 0
    
    
    likesDataWithAge = userDF.likesData
    
    #vectorizer = HashingVectorizer(non_negative=True)
    userAge = pd.merge(userAge,likesDataWithAge,how='right',on='userid')
    
    X = py.array(userAge.like_id).T
    X = X.astype(py.str)
    X = vectorizer.transform(X)
    
    prediction = clf.predict(X)
    print(prediction)
    
    userAge['age_group'] = prediction.astype(py.float)
    
    
    
    for l,group in userAge.groupby('userid'):
        #print(userDF.userData.loc[userDF.userData['userid']==l,'gender'])
        predictionResult = 1
        voterSize = 0
        for age_group, voter in group.groupby('age_group'):
                        
            if len(voter.index) > voterSize:
                predictionResult = age_group
                voterSize = len(voter.index)
        
        userDF.userData.loc[userDF.userData['userid']==l,'age_group'] = predictionResult          
       
         
    return
    

def LIWCPersonalitySVR(userDF,feature,cost,ga,selector):

    
    userDF.featureData.rename(columns={'userId':'userid'},inplace=True)
    LIWC = userDF.featureData
    personalityData = pd.merge(LIWC,userDF.userData,on='userid',how= 'right')
    
    svr_rbf = SVR(kernel='linear', C=cost,gamma=ga)
    linear_svr = LinearSVR()

    #neu origin 7932  0.18,0.0001: 7879 0.1,0.0001:7883 0.11,0.0001:7882

    

    X=personalityData.ix[:,'WC':'AllPct']
    X=selector.transform(X)
    print(X.shape)
    y=personalityData.ix[:,feature]
    svr_rbf.fit(X,y)
    
    return svr_rbf

    
    

def LIWCAgegroupSVM(userDF,selector):
    
    userDF.featureData.rename(columns={'userId':'userid'},inplace=True)
    userAge = userDF.userData
    userAge['age_group'] = userDF.userData['age'].apply(groupAge)
    LIWC = userDF.featureData
    userAge = pd.merge(LIWC,userAge,on='userid',how= 'right')
    
    svc_rbf = SVC(kernel = 'rbf',C = 1,
                  decision_function_shape='ovo',class_weight='balanced')

    #class_weight={1.0:1,2.0:0.5,3.0:0.2,4.0:0.1}
    
    X= userAge.ix[:,'WC':'AllPct']
    X= selector.transform(X)
    print(X.shape)
    y= userAge.ix[:,'age_group']
    
    svc_rbf.fit(X,y)    
    
    return svc_rbf

def LIWCGenderSVM(userDF,selector):
    
    userDF.featureData.rename(columns={'userId':'userid'},inplace=True)
    userGender = userDF.userData
    LIWC = userDF.featureData
    userGender = pd.merge(LIWC,userGender,on='userid',how= 'right')
    
    svc_rbf = SVC(kernel = 'rbf',class_weight='balanced',C=1)
    #gamma: 0.002,1-0.623 0.0003,1.5:0.650 0.0003,2:0.65(2.5:0.653)
    
    X= userGender.ix[:,'WC':'AllPct']
    X= selector.transform(X)
    print(X.shape)
    y= userGender.ix[:,'gender']
    
    svc_rbf.fit(X,y)    
    
    return svc_rbf

def LIWCGenderMNB(userDF,selector):

    userDF.featureData.rename(columns={'userId':'userid'},inplace=True)
    userGender = userDF.userData
    LIWC = userDF.featureData
    userGender = pd.merge(LIWC,userGender,on='userid',how= 'right')
    
    svc_MNB = MultinomialNB()
    #gamma: 0.002,1-0.623 0.0003,1.5:0.650 0.0003,2:0.65(2.5:0.653)
    
    X= userGender.ix[:,'WC':'AllPct']
    X= selector.transform(X)
    y= userGender.ix[:,'gender']
    
    svc_MNB.fit(X,y)    
    
    return svc_MNB
            