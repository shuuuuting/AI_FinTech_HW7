#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt

npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

trpn = trainface.shape[0]
trnn = trainnonface.shape[0]
tepn = testface.shape[0]
tenn = testnonface.shape[0]

#x跟y代表左上角起始點
#白色視窗寬高：w,h
#窮舉所有特徵
fn = 0
ftable = [] #用來記每個特徵的座標
#視窗1
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*2<=19): #座落在視窗內的就是合法特徵，記錄下來
                    fn = fn + 1
                    ftable.append([0,y,x,h,w])
#視窗2        
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w<=19): #座落在視窗內的就是合法特徵，記錄下來
                    fn = fn + 1
                    ftable.append([1,y,x,h,w])
#視窗3                   
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*3<=19): #座落在視窗內的就是合法特徵，記錄下來
                    fn = fn + 1
                    ftable.append([2,y,x,h,w])
#視窗4                    
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w*2<=19): #座落在視窗內的就是合法特徵，記錄下來
                    fn = fn + 1
                    ftable.append([3,y,x,h,w])                  

def fe(sample,ftable,c): #feature extraction
    ftype = ftable[c][0] #取第c個特徵
    y = ftable[c][1]
    x = ftable[c][2]
    h = ftable[c][3]
    w = ftable[c][4]
    T = np.arange(361).reshape((19,19)) #建0-360*0-360然後轉成19*19
    if(ftype==0): #白的減黑的
        idx1 = T[y:y+h,x:x+w].flatten() #白的index
        idx2 = T[y:y+h,x+w:x+w*2].flatten() #黑的index
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)
    elif(ftype==1): #白的減黑的
        idx1 = T[y:y+h*2,x:x+w].flatten() #白的index
        idx2 = T[y:y+h,x:x+w].flatten() #黑的index
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)
    elif(ftype==2): #白的減黑的
        idx1 = T[y:y+h,x:x+w].flatten() 
        idx2 = T[y:y+h,x+w:x+w*2].flatten() 
        idx3 = T[y:y+h,x+w*2:x+w*3].flatten() 
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)+np.sum(sample[:,idx3],axis=1)
    else: #白的減黑的
        idx1 = T[y:y+h,x:x+w].flatten() 
        idx2 = T[y:y+h,x+w:x+w*2].flatten() 
        idx3 = T[y+h:y+h*2,x:x+w].flatten()
        idx4 = T[y+h:y+h*2,x+w:x+w*2].flatten()
        output = np.sum(sample[:,idx1],axis=1)-np.sum(sample[:,idx2],axis=1)-np.sum(sample[:,idx3],axis=1)+np.sum(sample[:,idx4],axis=1)
    return output

trpf = np.zeros((trpn,fn)) #2429*36648(36648個特徵)
trnf = np.zeros((trnn,fn)) #4548*36648
tepf = np.zeros((tepn,fn)) 
tenf = np.zeros((tenn,fn))

for c in range(fn):
   trpf[:,c] = fe(trainface,ftable,c) 
   trnf[:,c] = fe(trainnonface,ftable,c)
   tepf[:,c] = fe(testface,ftable,c) 
   tenf[:,c] = fe(testnonface,ftable,c)

def WC(pw,nw,pf,nf): #弱分類器 ＃pw:positive sample權重，分對的調降，分錯的調升
    maxf = max(pf.max(),nf.max())
    minf = min(pf.min(),nf.min())
    theta = (maxf-minf)/10+minf
    error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
    polarity = 1
    if(error>0.5): #反指標
        polarity = 0
        error = 1-error
    min_theta = theta
    min_error = error
    min_polarity = polarity
    for i in range(2,10): #切九刀，找可以使錯誤率最低的那刀
        theta = (maxf-minf)*i/10+minf
        error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
        polarity = 1
        if(error>0.5): 
            polarity = 0
            error = 1-error
        if(error<min_error): 
            min_theta = theta
            min_error = error
            min_polarity = polarity
    return min_error,min_theta,min_polarity

amount = [1,3,5,20,100,200]
for n in amount:
    pw = np.ones((trpn,1))/trpn/2
    nw = np.ones((trnn,1))/trnn/2
    sc =[]
    for t in range(n): #取前幾大特徵
        weightsum = np.sum(pw)+np.sum(nw)
        pw = pw/weightsum #正規化
        nw = nw/weightsum
        best_error,best_theta,best_polarity = WC(pw,nw,trpf[:,0],trnf[:,0])
        best_feature = 0
        for i in range(1,fn):
            me,mt,mp = WC(pw,nw,trpf[:,i],trnf[:,i])
            if(me<best_error):
                best_error = me
                best_feature = i
                best_theta = mt
                best_polarity = mp
        beta = best_error/(1-best_error)
        if(best_polarity==1):
            pw[trpf[:,best_feature]>=best_theta]*=beta
            nw[trnf[:,best_feature]<best_theta]*=beta
        else:
            pw[trpf[:,best_feature]<best_theta]*=beta
            nw[trnf[:,best_feature]>=best_theta]*=beta
        alpha = np.log10(1/beta)
        sc.append([best_feature,best_theta,best_polarity,alpha])
        print(t)
        print(best_feature)
    
    trps = np.zeros((trpn,1))
    trns = np.zeros((trnn,1))
    teps = np.zeros((tepn,1))
    tens = np.zeros((tenn,1))
    alpha_sum = 0
    for i in range(n):
        feature = sc[i][0]
        theta = sc[i][1]
        polarity = sc[i][2]
        alpha_sum += alpha
        if(polarity==1):
            trps[trpf[:,feature]>=theta] += alpha
            trns[trnf[:,feature]>=theta] += alpha
            teps[tepf[:,feature]>=theta] += alpha
            tens[tenf[:,feature]>=theta] += alpha
        else:
            trps[trpf[:,feature]<theta] += alpha
            trns[trnf[:,feature]<theta] += alpha
            teps[tepf[:,feature]<theta] += alpha
            tens[tenf[:,feature]<theta] += alpha
    trps /= alpha_sum
    trns /= alpha_sum
    teps /= alpha_sum
    tens /= alpha_sum
    
    #ROC curve
    x = []
    y = []
    x_test = []
    y_test = []
    for i in range(1000):
        threshold = i/1000
        x.append(np.sum(trns>=threshold)/trnn)
        y.append(np.sum(trps>=threshold)/trpn)
        x_test.append(np.sum(tens>=threshold)/tenn)
        y_test.append(np.sum(teps>=threshold)/tepn)
    plt.title('train ROC')
    plt.plot(x,y,label='n = '+str(n))
    #plt.plot(x_test,y_test,label='test')
plt.legend()
plt.show()
'''    
for n in amount:
    sc_temp = sc[:n]
    teps = np.zeros((tepn,1))
    tens = np.zeros((tenn,1))
    trps = np.zeros((trpn,1))
    trns = np.zeros((trnn,1))
    alpha_sum = 0
    for i in range(n):
        feature = sc_temp[i][0]
        theta = sc_temp[i][1]
        polarity = sc_temp[i][2]
        alpha_sum += alpha
        if(polarity==1):
            teps[tepf[:,feature]>=theta] += alpha
            tens[tenf[:,feature]>=theta] += alpha
            trps[trpf[:,feature]>=theta] += alpha
            trns[trnf[:,feature]>=theta] += alpha
        else:
            teps[tepf[:,feature]<theta] += alpha
            tens[tenf[:,feature]<theta] += alpha
            trps[trpf[:,feature]<theta] += alpha
            trns[trnf[:,feature]<theta] += alpha
    teps /= alpha_sum
    tens /= alpha_sum
    trps /= alpha_sum
    trns /= alpha_sum
    x=[]
    y=[]
    x_test = []
    y_test = []
    for i in range(1000):
        threshold = i/1000
        x.append(np.sum(trns>=threshold)/trnn)
        y.append(np.sum(trps>=threshold)/trpn)
        x_test.append(np.sum(tens>=threshold)/tenn)
        y_test.append(np.sum(teps>=threshold)/tepn)
    plt.title(str(n)+' classifiers train ROC')
    plt.plot(x,y)
    plt.show()
    plt.title(str(n)+' classifiers test ROC')
    plt.plot(x_test,y_test)
    plt.show()
'''    
top100features = sc[:100]   
image = np.asarray(Image.open('17.JPG').convert('L'))
height,width = image.shape
position = []
#testimage = np.zeros((int(height*width/361),361)) 
#for i in range(int(height/19)):
#    for j in range(int(width/19)):
#        position.append([i*19,j*19]) #儲存切完19*19大小的小格後每個小格的左上角位置
testimage = np.zeros((int((height-19)*(width-19)/4),361)) 
for i in range(int((height-19)/2)): #一次移兩格
    for j in range(int((width-19)/2)):
        position.append([i*2,j*2]) #儲存切完19*19大小的小格後每個小格的左上角位置
for i in range(len(position)):
    pos_x = position[i][0] #左上角x座標
    pos_y = position[i][1] #左上角y座標
    grid = image[pos_x:pos_x+19,pos_y:pos_y+19] 
    testimage[i] = grid.flatten().reshape((1,361))
n = testimage.shape[0] 
f = np.zeros((n,fn))
for c in range(fn):
   print(c)
   f[:,c] = fe(testimage,ftable,c)
s = np.zeros((n,1))
alpha_sum = 0
face = []
for i in range(100):
    feature = sc[i][0]
    theta = sc[i][1]
    polarity = sc[i][2]
    alpha_sum += alpha
    if(polarity==1):
        s[f[:,feature]>=theta] += alpha
    else:
        s[f[:,feature]<theta] += alpha
s /= alpha_sum        
for i in range(len(s)):
    if(s[i]>0.53):
        face.append(position[i]) 

img = Image.open('17.JPG')
draw = ImageDraw.Draw(img)
for i in range(len(face)):
    x = face[i][1]
    y = face[i][0]
    draw.rectangle((x,y,x+19,y+19),outline='red')
img.show()

'''
me,mt,mp = WC(pw,nw,trpf[:,0],trnf[:,0])
mine = me
mini = 0
for i in range(1,fn):
    me,mt,mp = WC(pw,nw,trpf[:,0],trnf[:,0])
    if(me<mine):
        mine = me
        mini = i
print([mini,mine])        
'''