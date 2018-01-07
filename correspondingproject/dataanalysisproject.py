from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import scipy.stats




class docdata:
    
    fulldict = {}
    
    def __init__ (self):
        self.wordlist = []
        self.worddict = {}
        self.size = 0

def getdata(filenamelist):
    
    databasesize = 0
    datalist = []
    list_stopWords=list(set(stopwords.words('english')))
    delStr = str.maketrans(dict.fromkeys(string.punctuation + string.digits))
    
    wordlist = []
    
    for file in filenamelist:
        
        docfile = open(file,encoding='utf_8',errors='ignore')
        textstr = docfile.read()
        textstr = textstr.lower()
        textstr = textstr = textstr.translate(delStr)
        
        tokens = word_tokenize(textstr.strip())
        filtered = [w for w in tokens if w not in list_stopWords]
        print(filtered)
        wordlist.append(filtered)
    
    for doc in wordlist:
        data = docdata()
        data.size = len(doc)
        databasesize += data.size
        data.wordlist = doc
            
        for word in doc:
            
            if word in data.worddict.keys():
                data.worddict[word] = data.worddict[word] + 1            
            else:
                data.worddict[word] = 1
                
                
            if word in data.fulldict.keys():
                data.fulldict[word] = data.fulldict[word] +1
            else:
                data.fulldict[word] = 1
    
        datalist.append(data)
     
        
    return datalist

def buildtable(datalist):
    termdata = pd.DataFrame(columns = datalist[0].fulldict.keys())
    i = 0
    for data in datalist:
        for key in data.worddict.keys():
            termdata.loc[i,key] =  data.worddict[key]        
        i+=1
    print(termdata.shape)
    termdata.fillna(0,inplace=True)
    return termdata

def buildquerycolumn(datatable,query,w = 0):
    
    if w == 0:
        w = 1/len(query)
    
    datatable["keywordsdata"] = 0
    for key in query:
        key = key.lower()
        datatable.loc[:,"keywordsdata"] += w*datatable.loc[:,key]
        
    print (datatable["keywordsdata"])
    return
    

def analyzekeyword (filenamelist, query, rankthreshold=10):
    
    datatable = buildtable(getdata(filenamelist))
    buildquerycolumn(datatable, query)
    
    lowerquery = []
    for keyword in query:
        lowerquery.append(keyword.lower())
    lowerquery.append("keywordsdata")
    pvaluedict = {}
    
    for key in datatable.columns:
        
        
        if key not in lowerquery:
            pvaluedict[key] = scipy.stats.ttest_ind(datatable[key],datatable["keywordsdata"]).pvalue
 
    
    print(pvaluedict)
    sortedrankresult = sorted(pvaluedict.items(),key= lambda d:d[1],reverse = True)
    print(sortedrankresult)
    result = []
    
    for i in range(rankthreshold):
        result.append(sortedrankresult[i])
    print(result)
    return result


name = ['./sampleTop10/top1.txt','./sampleTop10/top2.txt','./sampleTop10/top3.txt','./sampleTop10/top4.txt']
query = ['Spain']
analyzekeyword(name, query)