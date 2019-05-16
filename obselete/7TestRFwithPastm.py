# Load libraries
import os
import ta
import warnings
import datetime
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

filename = os.path.join('data','BOM517044.csv')

def compute(filename,m,chp,est):
    st = datetime.datetime.now()
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
    df = df[prices+others]    
    df = df.iloc[::-1] # Reverse
    df = df.reset_index(drop=True) #Re-Index    
    print("==================================================================")        
    n = df.shape[0]
    
    y = []
    for i in range(m,n):    
        cprice = df[cpriceColumnName][i]
        oprice = df[opriceColumnName][i]                
        eff = cprice
        diff =  eff - oprice
        change = (diff*100)/oprice
        if eff > oprice and change > chp :
            y.append(1)
        else :
            y.append(0)    
            
    # Include feature(n) : for Last Sample (which is utmost needed)
    past_prices_list = []
    for i in range(m,n+1):
        past_prices_i = []
        base_i = i-m
        for j in range(m):
            for price in prices :
                past_prices_i.append(df[price][base_i+j])
        past_prices_list.append(past_prices_i)        
        
    # Feature Space 1
    x_price_cmp = []            
    for past_prices_i in past_prices_list :    
        cmp_prices_i = []
        cs = len(past_prices_i)
        for ci in range(0,cs-1) :
            for cj in range(ci+1,cs):                  
                if past_prices_i[ci] < past_prices_i[cj] :
                    cmp_prices_i.append(1)                
                elif past_prices_i[ci] == past_prices_i[cj] :
                    cmp_prices_i.append(5)                    
                else :
                    cmp_prices_i.append(9)                             
        x_price_cmp.append(cmp_prices_i)
        
    # Feature Space 2
    # Considering m = 2, talib values can be compared amongst themselves
    x_talib_stats = []    
    previous_columns = df.columns.values    
    talib = ta.add_all_ta_features(df,
                         opriceColumnName,
                         hpriceColumnName,
                         lpriceColumnName,
                         cpriceColumnName,
                         volumeColumnName,
                         fillna=True)  
    post_talib_rc = talib.shape[0]    
    talib = talib.drop(previous_columns,axis=1)    
    if post_talib_rc == n :
        for i in range(m,n+1):                     
            v = []
            base_i = i-m
            base_vals = talib.iloc[base_i].values
            for j in range(1,m):
                new_vals = talib.iloc[base_i+j].values
                new_vals_n = len(new_vals)
                for k in range(new_vals_n):
                    if base_vals[k] < new_vals[k] :
                        v.append(1)
                    elif base_vals[k] == new_vals[k] :
                        v.append(5)
                    else :
                        v.append(9)
                base_vals = new_vals
            x_talib_stats.append(v)
    else:
        print("Error At Talib")
    
    # Combining Feature Spaces
    x = []
    for i in range(m,n+1):
        base_i = i-m
        feature_i = x_price_cmp[base_i]
        feature_i.extend(x_talib_stats[base_i])
        x.append(feature_i)

    lastXN = len(x)-1
    lastFeature = x[lastXN] # The Enigma Key
    x = x[:lastXN] # Dropping Last Feature (Key)
    
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    
    datahealth = np.count_nonzero(y)/len(y)        
    
    fc = len(x[0])
    #print("Begin Features",fc)
    fc_names = []
    for fc_i in range(fc):
        fc_names.append(fc_i)          
    nfc_names = np.array(fc_names).astype(int)
    
    cur1 = 0
    cur2 = 0
    cur3 = 0
    best1 = 0    
    best2 = 0
    best3 = 0
    best_acc = 0
    best_prec = 0
    best_safe_acc = 0    
    best_feature_count = 0
    best_predictions = []
    pruned_features = []
    clf = None
    best_clf = None
    len_data = len(x)
    train_percentage = 75
    split_index = (len_data * train_percentage)//100
    ytrain = y[:split_index]
    ytests = y[split_index:]
    ntests = len(ytests)
    while len(pruned_features) < fc : 
        #print("Fc",fc,"Pruned",len(pruned_features))                   
        xn = np.delete(x, pruned_features, axis=1)        
        fn = np.delete(nfc_names, pruned_features, None)        
        
        xtrain = xn[:split_index]        
        xtests = xn[split_index:]        
                  
        clf=RandomForestClassifier(n_estimators=est,n_jobs=-1,random_state=1)        
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
        cur3 = fc-(len(pruned_features)-1)
        cur1 = metrics.precision_score(ytests, y_pred)
        cur2 = metrics.accuracy_score(ytests, y_pred)        
        cur_acc = metrics.accuracy_score(ytests, y_pred)    
        cur_prec = metrics.precision_score(ytests, y_pred)    
        mtn, mfp, mfn, mtp = metrics.confusion_matrix(ytests, y_pred).ravel()                                
        prisma1 = (cur1 > best1)
        prisma2 = (cur1==best1 and cur2 > best2)
        prisma3 = (cur1==best1 and cur2 == best2 and cur3 > best3)
        if  prisma1 or prisma2 or prisma3 :            
            best1 = cur1
            best2 = cur2
            best3 = cur3
            best_clf = clf
            best_acc = cur_acc            
            best_prec = cur_prec
            best_predictions = y_pred
            best_safe_acc = (ntests-mfp)/ntests            
            best_feature_count = fc-(len(pruned_features)-1)                        
            print("*")
            print("Precision  \t",cur1)
            print("'Accuracy  \t",cur2)
            print("'Features  \t",best_feature_count,"/",fc)
            

    y_pred = best_predictions            
    clf = best_clf  # Restoring Best Model    
    xbound = 200    # Limiting Pyplot    
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
    
    
    print("Health           \t",datahealth)    
    
    print("M                \t",m)
    print("Fc               \t",best_feature_count)    
    print("Chp              \t",chp)    
    print("Accuracy         \t",best_acc)
    print("Precision        \t",best_prec)
    print("Estimators       \t",est)
    #print("Safety Accuracy  \t",best_safe_acc)    
    et = datetime.datetime.now()
    tt = (et-st).seconds    
    print("\nTook",tt//60,"Mins",tt%60,"Seconds")
    
    
    
m = 4
chp = 2
#for m in range(4,5): # 4
    #for chp in range(2,5): # 2,3,4
        #compute(filename,m,chp)

for e in range(51,55,1):
    compute(filename,m,chp,e)
        
# Uptrend Precision from m = 3 -> 4
# Uptrend Accuracy with chp = 1 -> 5
# With m = 4 and chp : 2->3->4, model is reducing mfn ( so label health is -ve)
# Trying with more trees, accuracy decreasing strange at 300,500
# at 50 got 3 spikes with prec = 0 , acc = 0.8756
# default 100 is also doing decent with same accuracy
    

        























