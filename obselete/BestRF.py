# Load libraries
import os
import ta
import warnings
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt                              # analysis:ignore
from sklearn.ensemble import RandomForestClassifier

#%matplotlib qt
#%matplotlib inline
warnings.filterwarnings("ignore")

# 524500 KILITCH
# BOM517044 INTERNATION DATA MANAGEMENT

filename = os.path.join('datasets','BOM517044.csv')
m = 1
past = 1
init = m
chps = 1
chpi = 10 # 1 %
limit_chp = 30
debug = False
for m in range(init,past+1):
#for chpi in range(chps,limit_chp+1):
    df = pd.read_csv(filename)
    volumeColumnName = 'No. of Shares'
    opriceColumnName = 'Open'
    hpriceColumnName = 'High'
    lpriceColumnName = 'Low'
    cpriceColumnName = 'Close'    
    prices = [opriceColumnName,
              hpriceColumnName,
              lpriceColumnName,
              cpriceColumnName]
    others = [volumeColumnName]
    
    #prices = ['Open Price','High Price','Low Price','Close Price'] 
    df = df[prices+others]    
    df = df.iloc[::-1] # Reverse
    df = df.reset_index(drop=True) #Re-Index
    print("==================================================================")    
    #print("m",m)    
    #print("chp",chp)        
    t = []
    y = []
    for i in range(m,df.shape[0]):
        f = []
        z = i-m
        for j in range(m):
            for k in prices :
                f.append(df[k][z+j])
        t.append(f)    
        #aprice = df['Average Price'][i]        
        cprice = df[cpriceColumnName][i]
        oprice = df[opriceColumnName][i]        
        hprice = df[hpriceColumnName][i]
        eff = cprice
        diff =  eff - oprice
        change = (diff*100)/oprice
        if eff > oprice and change > (chpi/10) :
            y.append(1)
        else :
            y.append(0)    
    
    normalize = False
    x = []    
    for r in t :    
        cp = []
        cs = len(r)
        for ci in range(0,cs-1) :
            for cj in range(ci+1,cs):  
                '''
                a = r[ci]
                b = r[cj]
                percentage = ((a-b)*100)/b
                cp.append(percentage)                                             
                '''
                if r[ci] < r[cj] :
                    cp.append(1)                
                elif r[ci] == r[cj] :
                    cp.append(5)                    
                else :
                    cp.append(9)                             
        x.append(cp)
    
    price_fc = len(x[0])    
    pre_talib_rc = df.shape[0]
    previous_columns = df.columns.values
    '''       * TaLib *       '''
    kf = ta.add_all_ta_features(df,
                         opriceColumnName,
                         hpriceColumnName,
                         lpriceColumnName,
                         cpriceColumnName,
                         volumeColumnName,
                         fillna=True)  
    post_talib_rc = kf.shape[0]
    #print("Post Tabib Row Count",post_talib_rc)
    kf = kf.drop(previous_columns,axis=1)
    if post_talib_rc == pre_talib_rc :            
        rc = len(x)
        for xi in range(rc):            
            dfi = xi
            v = kf.iloc[dfi].values            
            x[xi].extend(v)
    else:
        print("*\n*\n*Talib Failure Check\n*\n*\n*")
    
    if debug:
        print("Price Fc         \t",price_fc)
        print("Extended Fc      \t",len(x[0]))        
    
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    
    datahealth = np.count_nonzero(y)/len(y)    
    
    if normalize :
        x = x/10 # Normalization :P
    
    fc = len(x[0])
    #print("Begin Features",fc)
    fc_names = []
    for fc_i in range(fc):
        fc_names.append(fc_i)          
    nfc_names = np.array(fc_names).astype(int)
    
    cur_acc = 0        
    best_acc = 0
    best_safe_acc = 0
    best_win_pred = 0
    best_feature_count = 0
    pruned_features = []
    clf = None
    best_clf = None
    while cur_acc >= best_acc and len(pruned_features) < fc :                    
        xn = np.delete(x, pruned_features, axis=1)        
        fn = np.delete(nfc_names, pruned_features, None)
        
        len_data = len(xn)
        train_percentage = 95
        split_index = (len_data * train_percentage)//100
        xtrain = xn[:split_index]
        ytrain = y[:split_index]
        xtests = xn[split_index:]
        ytests = y[split_index:]    
                  
        clf=RandomForestClassifier(n_estimators=1000,n_jobs=-1,random_state=1)        
        clf.fit(xtrain,ytrain)                
        feature_imp = pd.Series(clf.feature_importances_,index=fn).sort_values(ascending=False)        
        fk = int(feature_imp.tail(1).index[0])
        pruned_features.append(fk)
            
        '''
        sns.barplot(x=feature_imp, y=feature_imp.index)        
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.legend()
        plt.show()    
        '''
        
        y_pred=clf.predict(xtests)
    
        ntests = len(ytests)    
        cur_acc = metrics.accuracy_score(ytests, y_pred)    
        mtn, mfp, mfn, mtp = metrics.confusion_matrix(ytests, y_pred).ravel()                    
        if debug : 
            print("*\nAccuracy:",cur_acc)
            print("Safety Accuracy",(ntests-mfp)/ntests)
            print("Predicted Win Pct",mtp/(mtp + mfp))                  
    
        if cur_acc >= best_acc :
            best_clf = clf
            best_acc = cur_acc            
            best_safe_acc = (ntests-mfp)/ntests
            best_win_pred = mtp/(mtp + mfp)
            best_feature_count = fc-(len(pruned_features)-1)
            
                
    clf = best_clf # Restoring Best Model
    
    xbound = 100    # Limiting Pyplot
    
    
    limit = min(xbound,len(ytests))    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    #ax.set_xticks(np.arange(0, 50, 1))
    #ax.set_yticks(np.arange(0, 2, 0.1))
    plt.plot(ytests[0:limit],label='Actual')
    plt.plot(y_pred[0:limit],label='Predictions')
    plt.legend()
    plt.grid()
    plt.show()


    limit = min(xbound,len(ytests))
    safties = []
    for yi in range(limit):
        if y_pred[yi] == 1 and ytests[yi] == 0: #false Positives
            safties.append(0)
        else:
            safties.append(1)
            
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    #ax.set_xticks(np.arange(0, 50, 1))
    #ax.set_yticks(np.arange(0, 2, 0.1))
    plt.plot(safties,label='Safties')        
    plt.legend()
    plt.grid()
    plt.show()
    
    

    print("M                \t",m)
    print("Fc               \t",best_feature_count)    
    print("Chp              \t",chpi/10)
    print("Health           \t",datahealth)    
    print("Best Accuracy    \t",best_acc)
    print("Safety Accuracy  \t",best_safe_acc)
    print("Predicted Win Pct\t",best_win_pred)
    
    























