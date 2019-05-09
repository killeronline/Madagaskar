# Load libraries
import os
import ta
import sys
import talib
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
from sklearn.svm import SVC
warnings.filterwarnings("ignore")    

#%matplotlib qt

#def analysis(code,m,chp,est):
proceed = True
code = 'BOM532333'
m = 4
chp = 8
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
    # Cleaning Data
    # Remove All Non Traded Days ( Volume = 0 )
    df = df[df[volumeColumnName]>0]
    df = df.reset_index(drop=True)
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
        if eff > oprice and change > chp : # Extreme Positives
            y.append(1)
        elif eff < oprice and change < (-chp) : # Extreme Negatives
            y.append(0)
        else :
            y.append(-1) # Mixed Data Point        
            
        
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
    oldTalib = True
    if oldTalib :        
        x_talib_stats = []    
        previous_columns = df.columns.values    
        tadf = ta.add_all_ta_features(df,
                             opriceColumnName,
                             hpriceColumnName,
                             lpriceColumnName,
                             cpriceColumnName,
                             volumeColumnName,
                             fillna=True)  
        post_tadf_rc = tadf.shape[0]    
        tadf = tadf.drop(previous_columns,axis=1)    
        if post_tadf_rc == n :
            for i in range(m,n+1):                     
                v = []
                base_i = i-m
                base_vals = tadf.iloc[base_i].values
                for j in range(1,m):
                    new_vals = tadf.iloc[base_i+j].values
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
    else :
        x_talib_stats = []        
        previous_columns = df.columns.values    
        op = df['Open']
        hp = df['High']
        lp = df['Low']
        cp = df['Close']
        df['CDL2CROWS']=talib.CDL2CROWS(op,hp,lp,cp)
        df['CDL3BLACKCROWS']=talib.CDL3BLACKCROWS(op,hp,lp,cp)
        df['CDL3INSIDE']=talib.CDL3INSIDE(op,hp,lp,cp)
        df['CDL3LINESTRIKE']=talib.CDL3LINESTRIKE(op,hp,lp,cp)
        df['CDL3OUTSIDE']=talib.CDL3OUTSIDE(op,hp,lp,cp)
        df['CDL3STARSINSOUTH']=talib.CDL3STARSINSOUTH(op,hp,lp,cp)
        df['CDL3WHITESOLDIERS']=talib.CDL3WHITESOLDIERS(op,hp,lp,cp)
        df['CDLABANDONEDBABY']=talib.CDLABANDONEDBABY(op,hp,lp,cp)
        df['CDLADVANCEBLOCK']=talib.CDLADVANCEBLOCK(op,hp,lp,cp)
        df['CDLBELTHOLD']=talib.CDLBELTHOLD(op,hp,lp,cp)
        df['CDLBREAKAWAY']=talib.CDLBREAKAWAY(op,hp,lp,cp)
        df['CDLCLOSINGMARUBOZU']=talib.CDLCLOSINGMARUBOZU(op,hp,lp,cp)
        df['CDLCONCEALBABYSWALL']=talib.CDLCONCEALBABYSWALL(op,hp,lp,cp)
        df['CDLCOUNTERATTACK']=talib.CDLCOUNTERATTACK(op,hp,lp,cp)
        df['CDLDARKCLOUDCOVER']=talib.CDLDARKCLOUDCOVER(op,hp,lp,cp)
        df['CDLDOJI']=talib.CDLDOJI(op,hp,lp,cp)
        df['CDLDOJISTAR']=talib.CDLDOJISTAR(op,hp,lp,cp)
        df['CDLDRAGONFLYDOJI']=talib.CDLDRAGONFLYDOJI(op,hp,lp,cp)
        df['CDLENGULFING']=talib.CDLENGULFING(op,hp,lp,cp)
        df['CDLEVENINGDOJISTAR']=talib.CDLEVENINGDOJISTAR(op,hp,lp,cp)
        df['CDLEVENINGSTAR']=talib.CDLEVENINGSTAR(op,hp,lp,cp)
        df['CDLGAPSIDESIDEWHITE']=talib.CDLGAPSIDESIDEWHITE(op,hp,lp,cp)
        df['CDLGRAVESTONEDOJI']=talib.CDLGRAVESTONEDOJI(op,hp,lp,cp)
        df['CDLHAMMER']=talib.CDLHAMMER(op,hp,lp,cp)
        df['CDLHANGINGMAN']=talib.CDLHANGINGMAN(op,hp,lp,cp)
        df['CDLHARAMI']=talib.CDLHARAMI(op,hp,lp,cp)
        df['CDLHARAMICROSS']=talib.CDLHARAMICROSS(op,hp,lp,cp)
        df['CDLHIGHWAVE']=talib.CDLHIGHWAVE(op,hp,lp,cp)
        df['CDLHIKKAKE']=talib.CDLHIKKAKE(op,hp,lp,cp)
        df['CDLHIKKAKEMOD']=talib.CDLHIKKAKEMOD(op,hp,lp,cp)
        df['CDLHOMINGPIGEON']=talib.CDLHOMINGPIGEON(op,hp,lp,cp)
        df['CDLIDENTICAL3CROWS']=talib.CDLIDENTICAL3CROWS(op,hp,lp,cp)
        df['CDLINNECK']=talib.CDLINNECK(op,hp,lp,cp)
        df['CDLINVERTEDHAMMER']=talib.CDLINVERTEDHAMMER(op,hp,lp,cp)
        df['CDLKICKING']=talib.CDLKICKING(op,hp,lp,cp)
        df['CDLKICKINGBYLENGTH']=talib.CDLKICKINGBYLENGTH(op,hp,lp,cp)
        df['CDLLADDERBOTTOM']=talib.CDLLADDERBOTTOM(op,hp,lp,cp)
        df['CDLLONGLEGGEDDOJI']=talib.CDLLONGLEGGEDDOJI(op,hp,lp,cp)
        df['CDLLONGLINE']=talib.CDLLONGLINE(op,hp,lp,cp)
        df['CDLMARUBOZU']=talib.CDLMARUBOZU(op,hp,lp,cp)
        df['CDLMATCHINGLOW']=talib.CDLMATCHINGLOW(op,hp,lp,cp)
        df['CDLMATHOLD']=talib.CDLMATHOLD(op,hp,lp,cp)
        df['CDLMORNINGDOJISTAR']=talib.CDLMORNINGDOJISTAR(op,hp,lp,cp)
        df['CDLMORNINGSTAR']=talib.CDLMORNINGSTAR(op,hp,lp,cp)
        df['CDLONNECK']=talib.CDLONNECK(op,hp,lp,cp)
        df['CDLPIERCING']=talib.CDLPIERCING(op,hp,lp,cp)
        df['CDLRICKSHAWMAN']=talib.CDLRICKSHAWMAN(op,hp,lp,cp)
        df['CDLRISEFALL3METHODS']=talib.CDLRISEFALL3METHODS(op,hp,lp,cp)
        df['CDLSEPARATINGLINES']=talib.CDLSEPARATINGLINES(op,hp,lp,cp)
        df['CDLSHOOTINGSTAR']=talib.CDLSHOOTINGSTAR(op,hp,lp,cp)
        df['CDLSHORTLINE']=talib.CDLSHORTLINE(op,hp,lp,cp)
        df['CDLSPINNINGTOP']=talib.CDLSPINNINGTOP(op,hp,lp,cp)
        df['CDLSTALLEDPATTERN']=talib.CDLSTALLEDPATTERN(op,hp,lp,cp)
        df['CDLSTICKSANDWICH']=talib.CDLSTICKSANDWICH(op,hp,lp,cp)
        df['CDLTAKURI']=talib.CDLTAKURI(op,hp,lp,cp)
        df['CDLTASUKIGAP']=talib.CDLTASUKIGAP(op,hp,lp,cp)
        df['CDLTHRUSTING']=talib.CDLTHRUSTING(op,hp,lp,cp)
        df['CDLTRISTAR']=talib.CDLTRISTAR(op,hp,lp,cp)
        df['CDLUNIQUE3RIVER']=talib.CDLUNIQUE3RIVER(op,hp,lp,cp)
        df['CDLUPSIDEGAP2CROWS']=talib.CDLUPSIDEGAP2CROWS(op,hp,lp,cp)
        df['CDLXSIDEGAP3METHODS']=talib.CDLXSIDEGAP3METHODS(op,hp,lp,cp)
        post_rc = df.shape[0]    
        df = df.drop(previous_columns,axis=1)
        if post_rc == n :
            for i in range(m,n+1):
                base_i = i-m
                new_vals = df.iloc[base_i].values
                x_talib_stats.append(new_vals)
        else :
            print('Error At New Talib')
        
        
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

    # Considering Only Extremes
    # we have x and y here 

    lenY= len(y)
    y_sigma = []
    for i in range(lenY):
        y_sigma.append(y[i])    
        
    y_sigma.sort()
    lenYS = len(y_sigma)
    cutIndex = (lenYS*45)//100
    failCut = y_sigma[cutIndex]
    passCut = y_sigma[lenYS-cutIndex]
    print('FailCut       \t',failCut)
    print('PassCut       \t',passCut)
    
    extreme_x = []
    extreme_y = []
    lenNY = len(y)
    for i in range(lenNY):
        if y[i] >= 0 :
            extreme_x.append(x[i])
            extreme_y.append(y[i])
        '''
        elif y[i] < failCut :
            extreme_x.append(x[i])
            extreme_y.append(0)        
        '''
            
    x = extreme_x
    y = extreme_y
    
    
    
    print('Started Grids')    
    st = datetime.datetime.now()
    len_data = len(x)
    train_percentage = 50
    split_index = (len_data * train_percentage)//100
    xtrain = x[:split_index]
    ytrain = y[:split_index]
    xtests = x[split_index:]
    ytests = y[split_index:]

    classbalance_y = sum(y)/len(y)
    classbalance_ytrain = sum(ytrain)/len(ytrain)
    classbalance_ytests = sum(ytests)/len(ytests)
    
    print("balance y          \t",classbalance_y)
    print("balance ytrain     \t",classbalance_ytrain)
    print("balance ytests     \t",classbalance_ytests)
    
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
    # min_impurity_decrease= default best
    fc = len(xtrain[0])    
    # %matplotlib qt    
    # cwt = balanced and balanced_subsample are both good equally    
    #print("Begin Features",fc)
    fc_names = []
    for fc_i in range(fc):
        fc_names.append(fc_i)          
    nfc_names = np.array(fc_names).astype(int)
    
    pruned_features = []    
    while len(pruned_features) < fc//4 :        
    #for ichmoku in range(1,2):
        print( fc-len(pruned_features),'/',fc)
        
        xn = np.delete(x, pruned_features, axis=1)
        fn = np.delete(nfc_names, pruned_features, None)
        
        xtrain = xn[:split_index]        
        xtests = xn[split_index:]
        
        st = datetime.datetime.now()                        
            
        rfc = RandomForestClassifier(n_estimators=1000,
                                     class_weight='balanced',
                                     criterion='gini',
                                     random_state=1,
                                     verbose=False,
                                     n_jobs=-1)
            

      
        
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
        
        # Pruning feature space                
        cpfc = len(pruned_features)
        feature_imp = pd.Series(rfc.feature_importances_,index=fn).sort_values(ascending=False)        
        prunespeed = 10
        if (fc-cpfc) > prunespeed :
            fkmore = feature_imp.tail(prunespeed).index
            for fkpi in range(prunespeed):                
                fk = int(fkmore[fkpi])
                pruned_features.append(fk)
        else :
            break        
    
    plt.figure(figsize=(20, 16))    
    plt.plot(metro1,label='F1')    
    plt.plot(metro2,label='Recall')
    plt.plot(metro3,label='Accuracy')
    plt.plot(metro4,label='Precision')
    plt.legend()
    plt.grid()
    plt.show()    
        
    






















