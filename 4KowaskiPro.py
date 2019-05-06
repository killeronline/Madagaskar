# Load libraries
import os
import ta
import warnings
import datetime
import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")    

#%matplotlib qt

#def analysis(code,m,chp,est):
proceed = True
code = 'BOM503816'
m = 4
chp = 2
est = 100
#def analysis(code,m,chp,est,split):
if proceed:
    st = datetime.datetime.now()    
    database_path=os.path.join('database','main.db')
    conn=sql.connect(database_path)    
    df = pd.read_sql_query('select * from '+code, conn)    
    conn.close()
    volumeColumnName = 'No. of Shares'
    opriceColumnName = 'Open'
    hpriceColumnName = 'High'
    lpriceColumnName = 'Low'
    cpriceColumnName = 'Close' 
    ddates = ['Date']
    prices = [opriceColumnName,
              hpriceColumnName,
              lpriceColumnName,
              cpriceColumnName]
    others = [volumeColumnName]    
    df = df[ddates+prices+others]        
    print("==================================================================")        
    n = df.shape[0]
    last_date = df['Date'][n-1]
    last_close= df['Close'][n-1]
    last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()    
    last_date_str = last_date.strftime("%d %b %Y")
    df = df[prices+others]
    
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
    # Must be near to 0.5 ideally        
    databalance = sum(y)/len(y)
        
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
        
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)    
    
    # Remove Some Data (Ancestors) Which could be monotonous
    # like SMA(30) and EMA(20) which dont have good values till index30
    #ancestors = 30    
    #x = x[ancestors:]
    #y = y[ancestors:]    
    
    lastXN = len(x)-1
    lastFeature = x[lastXN] # The Enigma Key
    lastFeature = lastFeature.reshape(1,len(lastFeature))
    x = x[:lastXN] # Dropping Last Feature (Key)                    

    # GRID SEARCH    
    # we have x and y here 
    print('Started Grids')    
    st = datetime.datetime.now()
    '''
    gsc = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid={
            'max_depth': range(3,9),
            'n_estimators': (10, 50, 100, 500, 1000),
        },
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=0,              
        n_jobs=-1)
    grid_result = gsc.fit(x, y)
    best_params = grid_result.best_params_
    rfr = RandomForestClassifier(max_depth=best_params["max_depth"],
                                 n_estimators=best_params["n_estimators"],
                                 random_state=False,
                                 verbose=False)
    '''
    len_data = len(x)
    train_percentage = 50
    split_index = (len_data * train_percentage)//100
    xtrain = x[:split_index]
    ytrain = y[:split_index]
    xtests = x[split_index:]
    ytests = y[split_index:]        
    
    metro1 = []    
    metro2 = []    
    metro3 = []    
    metro4 = []    
    depth = 20
    min_samp_split = 4/100#0.04
    max_nodes = 30
    msLP = 0.011
    esti = 400
    #min_samp_leaf = 5 _ Dubious
    # OOB_SCORE doesnt matter
    # bootstrap=True
    # WarmStart doesnt matter
    
    fc = len(xtrain[0])    
    # %matplotlib qt    
    # cwt = balanced and balanced_subsample are both good equally
    
    for esti in range(1,2):
        
        st = datetime.datetime.now()        
        
        FirstChoice = True
        
        if FirstChoice :            
            rfc = RandomForestClassifier(n_estimators=1000,
                                         n_jobs=-1,
                                         criterion='gini',
                                         random_state=1)
            
        else :                    
            rfc = RandomForestClassifier(n_estimators=400,
                                         random_state=1,
                                         max_depth=depth,
                                         criterion='gini',
                                         min_samples_split=0.04,
                                         min_samples_leaf=0.011,
                                         max_features='auto',
                                         max_leaf_nodes=30,                                     
                                         class_weight='balanced',
                                         verbose=False)         
            
        print("Esti",esti)
        
        rfc.fit(xtrain,ytrain)
        y_pred=rfc.predict(xtests)
        #ytests = ytrain
        
        et = datetime.datetime.now()
        tt = (et-st).seconds
        timetaken = str(tt//60)+' Mins '+str(tt%60)+' Seconds'
        print('\n'+timetaken)    
        
        f1avg = 'binary'
        
        cur_acc = metrics.accuracy_score(ytests, y_pred)    
        cur_prec = metrics.precision_score(ytests, y_pred)    
        cur_recall = metrics.recall_score(ytests, y_pred)    
        cur_f1score = metrics.f1_score(ytests, y_pred, average=f1avg)
    
    
        print("*")
        print("F1                  \t",cur_f1score)
        print("Recall              \t",cur_recall)
        print("Accuracy            \t",cur_acc)
        print("Precision           \t",cur_prec)
        print("With Split Percent  \t",train_percentage)
        metro1.append(cur_f1score)
        metro2.append(cur_recall)
        metro3.append(cur_acc)
        metro4.append(cur_prec)
    
    
    plt.figure(figsize=(20, 16))    
    plt.plot(metro1,label='F1')    
    plt.plot(metro2,label='Recall')
    plt.plot(metro3,label='Accuracy')
    plt.plot(metro4,label='Precision')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    
    
    
    '''
    
    xlow = 0    
    xbound = 9999            
    limit = min(xbound,len(ytests))    
    fig = plt.figure(figsize=(20, 16))
    ax = fig.gca()
    #ax.set_xticks(np.arange(0, 50, 1))
    #ax.set_yticks(np.arange(0, 2, 0.1))
    plt.plot(ytests[xlow:limit],label='Actual')
    plt.plot(y_pred[xlow:limit],label='Predictions')
    plt.legend()
    plt.grid()
    plt.show()
    
    limit = min(xbound,len(ytests))
    safties = []
    for yi in range(xlow,limit):
        if y_pred[yi] == 1 and ytests[yi] == 0: #false Positives
            safties.append(0)
        else:
            safties.append(1)
            
    fig = plt.figure(figsize=(20, 16))
    ax = fig.gca()
    #ax.set_xticks(np.arange(0, 50, 1))
    #ax.set_yticks(np.arange(0, 2, 0.1))
    plt.plot(safties,label='Safties')        
    plt.legend()
    plt.grid()
    plt.show()
    '''
    
    
    '''
    fc = len(x[0])
    #print("Begin Features",fc)
    fc_names = []
    for fc_i in range(fc):
        fc_names.append(fc_i)          
    nfc_names = np.array(fc_names).astype(int)
    
    cur1 = 0
    cur2 = 0
    cur9 = 0    
    best1 = 0    
    best2 = 0
    best9 = 0    
    best_acc = 0
    best_prec = 0    
    best_recall = 0
    best_f1score = 0
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
    
    
    databalance_train = sum(ytrain)/len(ytrain)
    databalance_tests = sum(ytests)/len(ytests)
    yfinal = 0  # fail safe declaration
    ybear= 0    # sluggish
    ybull= 0    # agreesive
    while len(pruned_features) < fc : 
        #print("Fc",fc,"Pruned",len(pruned_features))                   
        xn = np.delete(x, pruned_features, axis=1)        
        fn = np.delete(nfc_names, pruned_features, None)        
        
        xtrain = xn[:split_index]        
        xtests = xn[split_index:]        
                  
        clf=RandomForestClassifier(n_estimators=est,
                                   n_jobs=-1,
                                   criterion='gini',
                                   random_state=1)        
        clf.fit(xtrain,ytrain)                        
        y_pred=clf.predict(xtests)                
        # cur1 and cur2 can be changed to find optimal models
        f1avg = 'binary'
        cur9 = len(pruned_features)
        cpfc = len(pruned_features)
        cur1 = metrics.precision_score(ytests, y_pred)        
        #cur2 = metrics.f1_score(ytests, y_pred, average=f1avg)        
        cur2 = metrics.accuracy_score(ytests, y_pred)    
        cur_acc = metrics.accuracy_score(ytests, y_pred)    
        cur_prec = metrics.precision_score(ytests, y_pred)    
        cur_recall = metrics.recall_score(ytests, y_pred)    
        cur_f1score = metrics.f1_score(ytests, y_pred, average=f1avg)
        mtn, mfp, mfn, mtp = metrics.confusion_matrix(ytests, y_pred).ravel()                                
        prisma1 = (cur1 > best1)
        prisma2 = (cur1==best1 and cur2 > best2)
        prisma3 = (cur1==best1 and cur2 == best2 and cur9 > best9)
        if  prisma1 or prisma2 or prisma3 :            
            best1 = cur1
            best2 = cur2
            best9 = cur9            
            best_clf = clf
            best_acc = cur_acc            
            best_prec = cur_prec            
            best_recall = cur_recall
            best_f1score = cur_f1score
            best_predictions = y_pred           
            best_feature_count = fc-cpfc                        
            
            # Making Revised Prediction            
            xfinal = lastFeature            
            if cpfc > 0 :                
                xfinal = np.delete(xfinal, pruned_features, axis=1)
                            
            yfinal=clf.predict(xfinal)
            yprobs=clf.predict_proba(xfinal)
            ybear= yprobs[0][0]
            ybull= yprobs[0][1]                        
            
        # Pruning feature space
        feature_imp = pd.Series(clf.feature_importances_,index=fn).sort_values(ascending=False)        
        prunespeed = 1
        if (fc-cpfc) > prunespeed :
            fkmore = feature_imp.tail(prunespeed).index
            for fkpi in range(prunespeed):                
                fk = int(fkmore[fkpi])
                pruned_features.append(fk)
        else :
            break
            

    y_pred = best_predictions            
    clf = best_clf  # Restoring Best Model    
    print("M                   \t",m)    
    print("Chp                 \t",chp)    
    print("F1                  \t",best_f1score)
    print("Recall              \t",best_recall)
    print("Accuracy            \t",best_acc)
    print("Precision           \t",best_prec)
    print("Estimators          \t",est)    
    print("Company Code        \t",code)        
    print("Feature Count       \t",best_feature_count)    
    print("Data Balance All    \t",databalance)
    print("Data Balance Train  \t",databalance_train)
    print("Data Balance Tests  \t",databalance_tests)
    et = datetime.datetime.now()
    tt = (et-st).seconds    
    timetaken = str(tt//60)+' Mins '+str(tt%60)+' Seconds'
    print('\n'+timetaken)    

    '''


#for i in range(30,90,5):
#analysis(code,m,chp,est,i)
    
        
    






















