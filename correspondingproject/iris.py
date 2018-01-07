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

