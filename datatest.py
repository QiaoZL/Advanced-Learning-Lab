from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel

from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from sklearn import linear_model

import numpy as np
from math import log
from src import LinearCombination

import sklearn_relief as relief
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFwe

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from skfeature.function.similarity_based import lap_score

import time
import sklearn
from bokeh.driving import linear

def calAIC(estimator, X, y):
        
    y_hat = estimator.predict(X)
    resid = y - y_hat
    sse = sum(resid**2)
    k = len(X[0])
    AIC= 2*k - 2*log(sse)
    
    return AIC


def Reliefseries():
    # Relief like algorithm
    
    #init
    n_iter = 100
    n_feat = 120
    Rf = relief.Relief()
    Rf.n_features = n_feat
    Rf.n_iterations = 100
      
    RfF = relief.ReliefF()
    RfF.n_features = n_feat
    RfF.n_iterations = 100
      
    RRfF = relief.RReliefF()
    RRfF.n_features = n_feat
    RRfF.n_iterations = 100
    #init done
      
    print(time.asctime(time.localtime(time.time())))
    print("relief select begin")
    # Rf_selected = Rf.fit_transform(X, y)
    print("Rf",Rf.random_state)
    Rf_selected_trans = Rf.fit_transform(X_transed, y)
    print("RfF",RfF.random_state)
    RfF_selected_trans = RfF.fit_transform(X_transed, y)
    print("RRfF",RRfF.random_state)
    RRfF_selected_trans = RRfF.fit_transform(X_transed, y)
    print("relief select done")
    print(time.asctime(time.localtime(time.time())))
      
    print('relief')
    Rf_result = cross_val_score(cls, X, y, cv = 5, scoring = 'accuracy' )
    print(Rf_result, sum(Rf_result)/5)
    Rf_result_trans = cross_val_score(cls, Rf_selected_trans, y, cv = 5, scoring = 'accuracy' )
    print(Rf_result_trans, sum(Rf_result_trans)/5)
      
    print('reliefF')
    RfF_result = cross_val_score(cls, X, y, cv = 5, scoring = 'accuracy' )
    print(RfF_result, sum(RfF_result)/5)
    RfF_result_trans = cross_val_score(cls, RfF_selected_trans, y, cv = 5, scoring = 'accuracy' )
    print(RfF_result_trans, sum(RfF_result_trans)/5)
       
    print('RreliefF')
    RRfF_result = cross_val_score(cls, X, y, cv = 5, scoring = 'accuracy' )
    print(RRfF_result, sum(RRfF_result)/5)
    RRfF_result_trans = cross_val_score(cls, RRfF_selected_trans, y, cv = 5, scoring = 'accuracy' )
    print(RRfF_result_trans, sum(RRfF_result_trans)/5)
      
    # Relief like algorithm done
    
    return

def Scoreseries():
    
    # Score like algorithm   
    # init
    n = 120
    test_alpha = 0.325
    
    f_features = SelectKBest(f_classif,k = n).fit_transform(X_transed,y)
    mi_features = SelectKBest(mutual_info_classif,k = n).fit_transform(X_transed,y)

    lap_featureindex = lap_score.lap_score(X_transed,y)
    lap_features = X_transed[:,lap_featureindex[0:n]]
    
    
    fdr_features = SelectFdr( alpha = 0.335).fit_transform(X_transed, y)
    print("fdr_features shape:",fdr_features.shape)
    fpr_features = SelectFpr( alpha = 0.33).fit_transform(X_transed, y)
    print("fpr_features shape:",fpr_features.shape)
    fwe_features = SelectFwe( alpha = test_alpha).fit_transform(X_transed, y)
    print("fwe_features shape:",fwe_features.shape)    

    baseresult = cross_val_score(cls, X, y, cv = 5, scoring = 'accuracy' )
                                               
#     chi2result = cross_val_score(cls, chi2_features, y, cv = 5, scoring = 'accuracy' )
#     print(baseresult,sum(baseresult)/5)
#     print(chi2result,sum(chi2result)/5)
    
    print("f")
    fresult = cross_val_score(cls, f_features, y, cv = 5, scoring = 'accuracy' )
    print(baseresult,sum(baseresult)/5)
    print(fresult,sum(fresult)/5)
    print("mutual information")
    miresult = cross_val_score(cls, mi_features, y, cv = 5, scoring = 'accuracy' )
    print(baseresult,sum(baseresult)/5)
    print(miresult,sum(miresult)/5)
    print("lap score")
    lapresult = cross_val_score(cls, lap_features, y, cv = 5, scoring = 'accuracy' )
    print(baseresult,sum(baseresult)/5)
    print(lapresult,sum(lapresult)/5)
    print("fdr")
    if fdr_features.shape[1] > 0:
        fdrresult = cross_val_score(cls, fdr_features, y, cv = 5, scoring = 'accuracy' )
        print(baseresult,sum(baseresult)/5)
        print(fdrresult,sum(fdrresult)/5)
        print("fpr")
    if fpr_features.shape[1] > 0:
        fprresult = cross_val_score(cls, fpr_features, y, cv = 5, scoring = 'accuracy' )
        print(baseresult,sum(baseresult)/5)
        print(fprresult,sum(fprresult)/5)
    if fwe_features.shape[1] > 0:
        print("fwe")
        fweresult = cross_val_score(cls, fwe_features, y, cv = 5, scoring = 'accuracy' )
        print(baseresult,sum(baseresult)/5)
        print(fweresult,sum(fweresult)/5)
    
    
    
        
    return

def regressionseries():

    thres = 'mean'
    mink = 5 #int(2/3 * all_features)
    
    lars_cv = linear_model.LassoLarsCV(cv=6).fit(X_transed,y)
    lars_features = SelectFromModel(lars_cv,prefit=True).transform(X_transed)
    print("lars_features shape:",lars_features.shape)
     
    f_features = SelectKBest(f_regression,k = max(mink,lars_features.shape[1])).fit_transform(X_transed,y)
    print("f_features shape:",f_features.shape)
    
    mi_features = SelectKBest(mutual_info_regression,k = max(mink,lars_features.shape[1])).fit_transform(X_transed,y)
    print("mi_features shape:",mi_features.shape)
    
    RRfF = relief.RReliefF()
    RRfF.n_features = max(mink,lars_features.shape[1])
    RRfF.n_iterations = 10
    RRfF_features = RRfF.fit_transform(X_transed, y) 
    print("RRfF_features shape:",RRfF_features.shape)
    
    test_alpha = 0.00005
    fdr_features = SelectFdr(score_func = f_regression, alpha = test_alpha).fit_transform(X_transed, y)
    print("fdr_features shape:",fdr_features.shape)
    fpr_features = SelectFpr(score_func = f_regression, alpha = test_alpha).fit_transform(X_transed, y)
    print("fpr_features shape:",fpr_features.shape)
    fwe_features = SelectFwe(score_func = f_regression, alpha = test_alpha).fit_transform(X_transed, y)
    print("fwe_features shape:",fwe_features.shape)
     
    r_baseresult = cross_val_score(rcls, X, y, cv = 5, scoring = 'neg_mean_squared_error' )
    print("base")
    print(r_baseresult,sum(r_baseresult)/5)
    lars_result = cross_val_score(rcls, lars_features, y, cv = 5, scoring = 'neg_mean_squared_error' )
    print("Lassocv")
    print(lars_result,sum(lars_result)/5)
    f_result = cross_val_score(rcls, f_features, y, cv = 5, scoring = 'neg_mean_squared_error' )
    print("f")
    print(f_result,sum(f_result)/5)
    mi_result = cross_val_score(rcls, mi_features, y, cv = 5, scoring = 'neg_mean_squared_error' )
    print("mi")
    print(mi_result,sum(mi_result)/5)
    RRfF_result = cross_val_score(rcls, RRfF_features, y, cv = 5, scoring = 'neg_mean_squared_error' )
    print("RRelief")
    print(RRfF_result,sum(RRfF_result)/5)
    fdr_result = cross_val_score(rcls, fdr_features, y, cv = 5, scoring = 'neg_mean_squared_error' )
    print("fdr")
    print(fdr_result,sum(fdr_result)/5)
    fpr_result = cross_val_score(rcls, fpr_features, y, cv = 5, scoring = 'neg_mean_squared_error' )
    print("fpr")
    print(fpr_result,sum(fdr_result)/5)
    fwe_result = cross_val_score(rcls, fwe_features, y, cv = 5, scoring = 'neg_mean_squared_error' )
    print("fwe")
    print(fwe_result,sum(fwe_result)/5)
    
    return


# multilabelprocess = LinearCombination.kernel_GramSchmidtProcess(rbf_kernel)
# data = np.random.random(size=[10,5])
# multilabelprocess.fit(data)
# print(multilabelprocess.kmatrix)
# print(multilabelprocess.basisweight)

# print(np.eye(3,3)[1])
# print(np.matrix(np.eye(3,3))[1])

# Xg,yg = datasets.make_gaussian_quantiles(n_features=10,random_state=42)
# print(Xg.shape,yg.shape)
# Xp,yp = datasets.make_multilabel_classification(n_samples=200,n_features=10,random_state=42,n_classes= 2)
# print(Xp.shape,yp.shape)
#  
# print(time.asctime(time.localtime(time.time())))
# multilabelprocess.fit(Xp)
# result = multilabelprocess.transform(Xp)
# print(result[0])
# print(Xp[0])
# print(multilabelprocess.kmatrix)
# print(multilabelprocess.basisweight)
# print(time.asctime(time.localtime(time.time())))


np.random.seed(42)

# classification problem
iris = datasets.load_iris()
print("iris",iris['data'].shape)
print(iris['target'].shape)
digit = datasets.load_digits(n_class=3)
print("digit",digit['data'].shape)
print(digit['target'].shape)
beast_cancer = datasets.load_breast_cancer()
print("beast_cancer",beast_cancer['data'].shape)
print(beast_cancer['target'].shape)
wine = datasets.load_wine()
print("wine",wine['data'].shape)
print(wine['target'].shape)

# regression problem
boston = datasets.load_boston()
print("boston",boston['data'].shape)
print(boston['target'].shape)
diabetes = datasets.load_diabetes()
print("diabetes",diabetes['data'].shape)
print(diabetes['target'].shape)
linnerud = datasets.load_linnerud()
print("linnerud",linnerud['data'].shape)
print(linnerud['target'].shape)
bodyfat = {}
bodyfat['data'],bodyfat['target'] = load_svmlight_file('datasets/bodyfat.txt')
print("bodyfat",bodyfat['data'].shape)
print(bodyfat['target'].shape)
eunite2001 = {}
eunite2001['data'],eunite2001['target'] = load_svmlight_file('datasets/eunite2001.txt')
print("eunite2001",eunite2001['data'].shape)
print(eunite2001['target'].shape)
mpg = {}
mpg['data'],mpg['target'] = load_svmlight_file('datasets/mpg.txt')
print("mpg",mpg['data'].shape)
print(mpg['target'].shape)
triazines = {}
triazines['data'],triazines['target'] = load_svmlight_file('datasets/triazines.txt')
print("triazines",triazines['data'].shape)
print(triazines['target'].shape)


simdata = datasets.make_regression()


cls = SVC()
rcls = LinearRegression()
is_regression = False

traindata = beast_cancer
all_features = traindata['data'].shape[1]

size = min(200,traindata['data'].shape[0])

X = traindata['data'][0:size]
y = traindata['target'][0:size]


scalerX = preprocessing.Normalizer(norm='l2')
standardScalerX = preprocessing.StandardScaler()
standardScalerY = preprocessing.StandardScaler()

print(time.asctime(time.localtime(time.time())))
print("trans begin")
KGSP = LinearCombination.kernel_GramSchmidtProcess(rbf_kernel)
KGSP.fit(X)
X_transed = KGSP.transform(X)
X_transed = standardScalerX.fit_transform(X_transed)
if is_regression:
    y = standardScalerY.fit_transform(y.reshape(-1,1))
    y = y.reshape(size,)
print("trans done")
# print(X_transed)
print(time.asctime(time.localtime(time.time())))


Reliefseries()
Scoreseries()
#regressionseries()





