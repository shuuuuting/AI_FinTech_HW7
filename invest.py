#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('TXF20112015.csv',sep=',',header=None)
TAIEX = df.values
tradeday = list(set(TAIEX[:,0]//10000))
tradeday.sort()

def show(profit):
    profit = np.array(profit)
    profit2 = np.cumsum(profit)
    ans1 = profit2[-1] #總損益點數
    ans2 = np.sum(profit>0)/len(profit) #勝率
    ans3 = np.mean(profit[profit>0])#賺錢時獲利點數
    ans4 = np.mean(profit[profit<=0])#輸錢時損失點數
    print('total:',ans1,'\nwin ratio:',ans2,'\nwin avg:',ans3,'\nlose avg:',ans4,'\nprofit distribution:')
    plt.hist(profit,bins=100)
    plt.show()
    print('accumulated profit:')
    plt.plot(profit2)
    plt.show()
    
#strategy 0.0 開盤買進一口台指期，收盤時平倉
profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    profit[i] = TAIEX[idx[-1],1] - TAIEX[idx[0],2]
print('************ strategy 0.0 ************')
show(profit)

#strategy0.1 開盤空一口台指期，收盤時平倉
profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    profit[i] = TAIEX[idx[0],2] - TAIEX[idx[-1],1]
print('************ strategy 0.1 ************')
show(profit)

#strategy1.0 開盤買進一口，30點停損，收盤平倉
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2] #開盤價買入
    idx2 = np.nonzero(TAIEX[idx,4]<=p1-30)[0] #idx是屬於今天的index，最低價拿出來 ＃idx2是一天的300分鐘內低於30點的分鐘
    if(len(idx2)==0):
        p2 = TAIEX[idx[-1],1] #沒有的話用平倉價買
    else:
        p2 = TAIEX[idx[idx2[0]],1] 
    profit.append(p2-p1)
print('************ strategy 1.0 ************')
show(profit)

#strategy1.1 開盤空一口，30點停損，收盤平倉
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2] 
    idx2 = np.nonzero(TAIEX[idx,4]>=p1+30)[0] 
    if(len(idx2)==0):
        p2 = TAIEX[idx[-1],1] 
    else:
        p2 = TAIEX[idx[idx2[0]],1] 
    profit.append(p1-p2)
print('************ strategy 1.1 ************')
show(profit)

#strategy2.0 開盤買進一口，30點停損，30點停利，收盤平倉
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2]
    idx2 = np.nonzero(TAIEX[idx,4]<=p1-30)[0] #停損
    idx3 = np.nonzero(TAIEX[idx,3]>=p1+30)[0] #停利
    if(len(idx2)==0 and len(idx3)==0):
        p2 = TAIEX[idx[-1],1]
    elif(len(idx3)==0): #沒有停利點只有停損點
        p2 = TAIEX[idx[idx2[0]],1] 
    elif(len(idx2)==0): #沒有停損點只有停利點
        p2 = TAIEX[idx[idx3[0]],1]
    #都有的話就比早，看停損停利誰比較早發生
    elif(idx2[0]<idx3[0]):
        p2 = TAIEX[idx[idx2[0]],1]
    else: 
        p2 = TAIEX[idx[idx3[0]],1] 
    profit.append(p2-p1)
print('************ strategy 2.0 ************')
show(profit)

#strategy2.1 開盤空一口，30點停損，30點停利，收盤平倉
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2]
    idx2 = np.nonzero(TAIEX[idx,4]>=p1+30)[0] #停損
    idx3 = np.nonzero(TAIEX[idx,3]<=p1-30)[0] #停利
    if(len(idx2)==0 and len(idx3)==0):
        p2 = TAIEX[idx[-1],1]
    elif(len(idx3)==0): #沒有停利點只有停損點
        p2 = TAIEX[idx[idx2[0]],1] 
    elif(len(idx2)==0): #沒有停損點只有停利點
        p2 = TAIEX[idx[idx3[0]],1]
    #都有的話就比早，看停損停利誰比較早發生
    elif(idx2[0]<idx3[0]):
        p2 = TAIEX[idx[idx2[0]],1]
    else: 
        p2 = TAIEX[idx[idx3[0]],1] 
    profit.append(p1-p2)
print('************ strategy 2.1 ************')
show(profit)

#strategy3.0
cumsum = []
a = []
for n in range(0,110,10):
    for m in range(n,110,10):
        profit= []
        for i in range(len(tradeday)):
            date = tradeday[i]
            idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
            idx.sort()
            p1 = TAIEX[idx[0],2]
            idx2 = np.nonzero(TAIEX[idx,4]<=p1-n)[0] #停損
            idx3 = np.nonzero(TAIEX[idx,3]>=p1+m)[0] #停利
            if(len(idx2)==0 and len(idx3)==0):
                p2 = TAIEX[idx[-1],1]
            elif len(idx3)==0:
                p2 = TAIEX[idx[idx2[0]],1]
            elif len(idx2)==0:
                p2 = TAIEX[idx[idx3[0]],1]
            elif idx2[0]<idx3[0]:
                p2 = TAIEX[idx[idx2[0]],1]
            else:
                p2 = TAIEX[idx[idx3[0]],1]
            profit.append(p2-p1)
        profit=np.array(profit)
        profit2 = np.cumsum(profit)
        cumsum.append([n,m,profit2[-1]])

c = np.array(cumsum)
d = np.where(c==np.max(c)) #取出最大值的索引
e = int(d[0])
profit= []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2]
    idx2 = np.nonzero(TAIEX[idx,4]<=p1-cumsum[e][0])[0]#停損
    idx3 = np.nonzero(TAIEX[idx,3]>=p1+cumsum[e][1])[0]#停利
    if(len(idx2)==0 and len(idx3)==0):
        p2 = TAIEX[idx[-1],1]
    elif len(idx3)==0:
        p2 = TAIEX[idx[idx2[0]],1]
    elif len(idx2)==0:
        p2 = TAIEX[idx[idx3[0]],1]
    elif idx2[0]<idx3[0]:
        p2 = TAIEX[idx[idx2[0]],1]
    else:
        p2 = TAIEX[idx[idx3[0]],1]
    profit.append(p2-p1)
print('************ strategy 3.0 ************')
show(profit)

#strategy3.1
cumsum = []
a = []
for n in range(0,110,10):
    for m in range(n,110,10):
        profit= []
        for i in range(len(tradeday)):
            date = tradeday[i]
            idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
            idx.sort()
            p1 = TAIEX[idx[0],2]
            idx2 = np.nonzero(TAIEX[idx,4]>=p1+n)[0] #停損
            idx3 = np.nonzero(TAIEX[idx,3]<=p1-m)[0] #停利
            if(len(idx2)==0 and len(idx3)==0):
                p2 = TAIEX[idx[-1],1]
            elif len(idx3)==0:
                p2 = TAIEX[idx[idx2[0]],1]
            elif len(idx2)==0:
                p2 = TAIEX[idx[idx3[0]],1]
            elif idx2[0]<idx3[0]:
                p2 = TAIEX[idx[idx2[0]],1]
            else:
                p2 = TAIEX[idx[idx3[0]],1]
            profit.append(p2-p1)
        profit=np.array(profit)
        profit2 = np.cumsum(profit)
        cumsum.append([n,m,profit2[-1]])

c = np.array(cumsum)
d = np.where(c==np.max(c)) #取出最大值的索引
e = int(d[0])
profit= []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2]
    idx2 = np.nonzero(TAIEX[idx,4]>=p1+cumsum[e][0])[0]#停損
    idx3 = np.nonzero(TAIEX[idx,3]<=p1-cumsum[e][1])[0]#停利
    if(len(idx2)==0 and len(idx3)==0):
        p2 = TAIEX[idx[-1],1]
    elif len(idx3)==0:
        p2 = TAIEX[idx[idx2[0]],1]
    elif len(idx2)==0:
        p2 = TAIEX[idx[idx3[0]],1]
    elif idx2[0]<idx3[0]:
        p2 = TAIEX[idx[idx2[0]],1]
    else:
        p2 = TAIEX[idx[idx3[0]],1]
    profit.append(p1-p2)
print('************ strategy 3.1 ************')
show(profit)