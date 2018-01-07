
import sys,csv
import pandas as pd
import numpy as np

from userData import user
from sklearn import datasets
from learningTools import groupAge

class userDataStruct():
    userData = ""
    likesData = ""
    featureData = ""
    textData = 0

def sampleInput(inputFile):
    
    inputFilePro = inputFile + "/profile/profile.csv"
    
    csvReader = csv.reader(open(inputFilePro))
    
    count = 0
    users = []
    
    for row in csvReader:
        
        if count == 0:
            count = count+1
            continue
        
        parameterSplit = ','.join(row)
        parameters = parameterSplit.split(',')
        
        singleUser = user(1)
        singleUser.id = parameters[1]
        singleUser.age = float(parameters[2])
        singleUser.genderType = float(parameters[3])
        singleUser.open = float(parameters[4])
        singleUser.conscientious = float(parameters[5])
        singleUser.extrovert = float(parameters[6])
        singleUser.agreeable = float(parameters[7])
        singleUser.neurotic = float(parameters[8])
                
        user.idList.append(parameters[1])
        users.append(singleUser)
        
        print(len(users))
    
    inputFileRel = inputFile + "/relation/relation.csv"
    
    csvReader = csv.reader(open(inputFileRel))
    count = 0
    userID = -1
    '''
    for row in csvReader:
        
        if count == 0:
            count = count+1
            continue       
        
        parameterSplit = ','.join(row)
        parameters = parameterSplit.split(',')
                               
        userID = parameters[1]
        print(parameters[1])
        users[user.idList.index(userID)].likeID.append(parameters[2])
    
    '''
    return users

def userInput(inputFile):
    
    inputFilePro = inputFile + "/profile/profile.csv"
    
    csvReader = csv.reader(open(inputFilePro))
    
    count = 0
    users = []
    
    for row in csvReader:
        
        if count == 0:
            count = count+1
            continue
        
        parameterSplit = ','.join(row)
        parameters = parameterSplit.split(',')
        
        singleUser = user(1)
        singleUser.id = parameters[1]
        user.idList.append(parameters[1])
        users.append(singleUser)
    
    
    inputFileRel = inputFile + "/relation/relation.csv"
    
    csvReader = csv.reader(open(inputFileRel))
    count = 0
    userID = -1
    for row in csvReader:
        
        if count == 0:
            count = count+1
            continue       
        
        parameterSplit = ','.join(row)
        parameters = parameterSplit.split(',')
                               
        userID = parameters[1]
        users[user.idList.index(userID)].likeID.append(parameters[2])
    
    return users

def sampleInputPd(inputFile):
    userDF = userDataStruct()
    userDF.userData = pd.read_csv(inputFile+"/profile/profile.csv")
    userDF.likesData = pd.read_csv(inputFile+"/relation/relation.csv")
    userDF.featureData = pd.read_csv(inputFile+"/LIWC/LIWC.csv")
    userDF.textData = datasets.load_files(inputFile,load_content=True,categories='text')
    
    return userDF

def likesDataprepocessing(userDF,likesize):
    
    #userDF.featureData.rename(columns={'userId':'userid'},inplace=True)  
    
    print(likesize)
    likesData = pd.merge(userDF.userData,userDF.likesData,on='userid',how= 'right')
    likesData = likesData.groupby('like_id').filter(
        lambda like_id: len(like_id) > likesize)
    likesData = likesData.reindex()
    #print(likesData.index)
    userlikegroup = likesData.groupby('like_id').groups
    useridgroup = likesData.groupby('userid').groups

    newfeature = pd.DataFrame(columns=userlikegroup.keys())
    newfeature['userid']=userDF.userData['userid']
    newfeature['gender']=userDF.userData['gender']
    newfeature['age_group']=userDF.userData['age'].apply(groupAge)

    newfeature = newfeature.fillna(0)
    

    #print(newfeature)

    #print(likesData.loc[0,['userid']])
    
    for key in useridgroup.keys():
        #print(key)
        iuserid = key
        tempindex= useridgroup[iuserid]
        target = likesData.ix[tempindex,['like_id']]#.reindex()
        #print(target)
        newfeature.ix[newfeature['userid'] == iuserid,target['like_id']]=1
        #print(newfeature.ix[newfeature['userid'] == iuserid,target['like_id']]) #replace(0,1,inplace=True)
    
    return newfeature

    
    
    
    