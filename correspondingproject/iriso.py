#################################################################
#
#  TCSS555 - HW 1
#
#  Martine De Cock
#
#################################################################

import pandas as pd
from sklearn import tree
from sklearn import metrics

# Read the dataset into a dataframe and map the labels to numbers
df = pd.read_csv('iris.csv')
map_to_int = {'setosa':0, 'versicolor':1, 'virginica':2}
df["label"] = df["species"].replace(map_to_int)
print(df)

# Separate the input features from the label
features = list(df.columns[:4])
X = df[features]
y = df["label"]

# Train a decision tree and compute its training accuracy
clf = tree.DecisionTreeClassifier(max_depth=2, criterion='entropy')
clf.fit(X, y)

print(metrics.accuracy_score(y,clf.predict(X)))

sumAcc = 0
size = len(df.index)
partSize = round(size/10)
originDF = df

for index in range(0,10):
    
    begin = index*partSize
    end = (index+1)*partSize
    
    partA = originDF[0:begin]
    partB = originDF[begin+1:end]
    partC = originDF[end+1:size-1]
    
    df = pd.concat([partA,partC])
# Separate the input features from the label
    features = list(df.columns[:4])
    X = df[features]
    y = df["label"]

# Train a decision tree and compute its training accuracy
    clf = tree.DecisionTreeClassifier(max_depth=2, criterion='entropy')
    clf.fit(X, y)
    
    X = partB[features]
    y = partB["label"]
    print(metrics.accuracy_score(y,clf.predict(X)))
    sumAcc = sumAcc + metrics.accuracy_score(y,clf.predict(X))

sumAcc = sumAcc/10
print("10-folds: ",sumAcc) 
