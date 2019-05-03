# Load libraries
import os
import ta
import Helpers
import warnings
import datetime
import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

def analysis(code,chp):    
    m = 1    
    debug = False    
    database_path=os.path.join('database','main.db')
    conn=sql.connect(database_path)    
    df = pd.read_sql_query('select * from '+code, conn)    
    conn.close()
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
    size = df.shape[0]
    last_date = df['Date'][size-1]
    last_close= df['Close'][size-1]
    last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()    
    last_date_str = last_date.strftime("%d %b %Y")
    print("==================================================================")            
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
        eff = cprice
        diff =  eff - oprice
        change = (diff*100)/oprice
        if eff > oprice and change > chp :
            y.append(1)
        else :
            y.append(0)    
        
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
    best_xtests_last = []
    pruned_features = []
    clf = None
    best_clf = None
    while cur_acc >= best_acc and len(pruned_features) < fc :                    
        xn = np.delete(x, pruned_features, axis=1)        
        fn = np.delete(nfc_names, pruned_features, None)
        
        len_data = len(xn)
        train_percentage = 95 # scope for improvement by tuning here
        split_index = (len_data * train_percentage)//100
        xtrain = xn[:split_index]
        ytrain = y[:split_index]
        xtests = xn[split_index:]
        ytests = y[split_index:]    
                  
        clf=RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=1)        
        clf.fit(xtrain,ytrain)                
        feature_imp = pd.Series(clf.feature_importances_,index=fn).sort_values(ascending=False)        
        fk = int(feature_imp.tail(1).index[0])
        pruned_features.append(fk)            
        
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
    y_pred=clf.predict(xtests)
    yfinal = 0
    return [chp,datahealth,best_acc,best_safe_acc,best_win_pred,yfinal]
    

# Main of Kowaski  
chp = 1
results = []
metadata = Helpers.MetaData()
codes_names = metadata.healthy_codes
for (code,name) in codes_names.items():
    result = analysis(code,chp)
    results.append(result)
    






















