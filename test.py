from asyncore import write
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from math import log

class Domain:
  def __init__(self,_name,_label,_length,_yuanyin,_entropy,_segmentation):
    self.name=_name
    self.label=_label
    self.length=_length
    self.yuanyin=_yuanyin
    self.segmentation=_segmentation
    self.entropy=_entropy

  def returnData(self):
    return [self.length, self.yuanyin,self.entropy,self.segmentation]

  def returnLabel(self):
    if(self.label=="dga"):
      return 0
    else:
      return 1

def yuanyinCal(strs):
  yuanyin_list=['a','e','i','o','u']
  count=0
  for ch in strs:
    if ch in yuanyin_list:
      count+=1
  return count/len(strs)

def InfoEntropy(str_1):
    Info_map = {}

    for i in str_1:
        #   不统计的字符↓
        if i != ' ' and i != '"' and i != "." and i != ',':
            if i in Info_map.keys():
                Info_map[i] += 1
            else:
                Info_map[i] = 1

    return calcShannonEnt(Info_map)


def calcShannonEnt(dataSet):
    numEntries = 0
    shannonEnt = 0.0

    for key in dataSet:
        numEntries += dataSet[key]

    # 计算信息熵
    for key in dataSet:
        prob = float(dataSet[key]) / numEntries  # 计算p(xi)
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt

def initData(filename,domainlist):
  with open(filename) as f:
    for line in f:
      line=line.strip()
      if line.startswith("#") or line=="":
        continue
      tokens=line.split(",")
      domain_name=tokens[0].split(".")
      name=domain_name[0]
      if len(tokens)==2:
        label=tokens[1]
      else:
        label="unknown"
      length=len(name)
      yuanyin=yuanyinCal(name)
      entropy=InfoEntropy(name)
      segmentation=len(domain_name)
      domainlist.append(Domain(name,label,length,yuanyin,entropy,segmentation))

def writeData(filename,resultList):
  with open(filename,'w+') as f:
    for item in resultList:
      if item==0:
        f.write("dga"+"\n")
      if item==1:
        f.write("notdga"+"\n")

def main():
  traindomainlist=[]
  testdomainlist=[]
  initData("train.txt",traindomainlist)
  initData("test.txt",testdomainlist)
  featureMatrix=[]
  labelList=[]
  testMatrix=[]
  for item in traindomainlist:
    featureMatrix.append(item.returnData())
    labelList.append(item.returnLabel())
  for item in testdomainlist:
    testMatrix.append(item.returnData())
  clf=RandomForestClassifier(random_state=0)
  clf.fit(featureMatrix,labelList)
  resultList=clf.predict(testMatrix)
  writeData("result.txt",resultList)
  
main()





   
