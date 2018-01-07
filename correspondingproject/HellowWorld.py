
import sys,getopt,random

from inputTools import userInput
from inputTools import sampleInput
from outputTools import outputResult
from learningTools import simpleLearning



if __name__ == '__main__':
    
    opts,args = getopt.getopt(sys.argv[1:], "i:o:")
    inputFile = ""
    outputFile = ""
    trainingFile = 'data/training'
    
    for op,value in opts:
        if op == "-i":
            inputFile = value
        if op == "-o":
            outputFile = value

            
    
    
    print(inputFile,outputFile,trainingFile)
    users = userInput(inputFile)
    samples = sampleInput(trainingFile) 
    print("Samples Done!")
    classifier = simpleLearning(samples)
    
    print("Classifier Done!")
    
    for user in users:
        user.age = classifier[0]
        if user.age <= 24:
            user.age_group = "xx-24"
        if user.age > 24 and user.age < 34:
            user.age_group = "25-34"
        if user.age > 34 and user.age <= 49:
            user.age_group = "35-49"
        if user.age > 49:
            user.age_group = "49-xx"
            
        user.genderType = classifier[1]
        if random(0,1) >= user.genderType:
            user.gender = "female"
        else:
            user.gender = "male"
        
        para = random(1,5)    
        user.open = classifier[2]*para/5
        user.conscientious = classifier[3]*para/5
        user.extrovert = classifier[4]*para/5
        user.agreeable = classifier[5]*para/5
        user.neurotic = classifier[6]*para/5
        
    
    
        outputResult(user,outputFile)
    
    
    pass

