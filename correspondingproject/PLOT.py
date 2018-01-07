import matplotlib.pyplot as plt

axes = plt.subplot(1,1,1)

plusX1 = [10,0,5]
plusX2 = [0,-10,-2]
minusX1 = [5,0,5]
minusX2 = [10,5,5]

lineX1 = [-1,11]
lineY1 = [1.5,1.5]

axes.plot(lineX1,lineY1,c="black",label = 'Decision Boundry')

axes.scatter(plusX1,plusX2,marker='*',color = 'red',s=300,label='+')
axes.scatter(minusX1,minusX2,marker='.',color = 'blue',s=300,label='-')
axes.legend()
plt.show()

#republican_train.csv republican_test.csv republican