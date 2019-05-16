# Load libraries
import os
import sys                                                    # analysis:ignore
import talib
import warnings
import datetime
import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")    

#%matplotlib qt

#def analysis(code,m,chp,est):
proceed = True

code = 'BOM500399'
m = 4
mz = 2
chp = 6
est = 100
split = 50

def analysis(code,m,mz,chp,est,split):
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
        
    '''
    n = 10 -> m = 4 -> 0 1 2 3 ... 4 5 6 7 ... 8 9 <- mz = 3
    i < n-2 end = n - mz + 1
    n-3 , n-2 , n-1 will be targets
    base_i = i-m
    base_z = i+mzi (0,1,...mz-1)    
    samples = n-m-mz+1
    '''
    y = []
    avg_closes = []
    init_opens = []
    conc_open_closes = []    
    for i in range(m,n-mz+1):
        oprice = df[opriceColumnName][i]
        mz_closes = []        
        for mzi in range(mz):            
            base_z = i + mzi
            cprice = df[cpriceColumnName][base_z]
            mz_closes.append(cprice)
                        
        conc_open_closes.append([oprice]+mz_closes)                
        avg_cprice = sum(mz_closes)/mz            
        avg_closes.append(avg_cprice)
        init_opens.append(oprice)
        eff = avg_cprice
        diff =  eff - oprice
        change = (diff*100)/oprice
        if eff > oprice and change > chp : # Extreme Positives
            y.append(1)
        elif eff < oprice and change < (-chp) : # Extreme Negatives
            y.append(0)
        else :
            y.append(-1) # Mixed Data Point        
        
    ''' Visualization    
    # 1, Plot Closes        
    '''    
    close = []
    for i in range(m,n-mz+1):
        cprice = df[cpriceColumnName][i]
        close.append(cprice)        
    
    close = np.array(close)
    y = np.array(y)
    
    close_p = np.ma.masked_where(~(y== 1) ,close)
    close_n = np.ma.masked_where(~(y== 0) ,close)
    close_e = np.ma.masked_where(~(y==-1) ,close)
    
    '''
    fig, ax = plt.subplots()        
    ax.plot(init_opens,'black')
    ax.plot(avg_closes,'blue')
    ax.plot(close,'grey')
    ax.plot(close_e,'cyan')
    ax.plot(close_p,'green',marker='^')
    ax.plot(close_n,'red')
        
    plt.show()
    '''
    
    
        
    # Feature Space 2
    # Considering m = 2, talib values can be compared amongst themselves
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
        print('Done Talib')
    else :
        print('Error At New Talib')

    # Sum of Talib Candles
    x_talib_cdl_sum = np.sum(df, axis = 1)
        
    
    # Combining Feature Spaces
    x = []
    for i in range(m,n-mz+1):      
        vals = []                
        vals.extend(df.iloc[i].values)        
        #vals.extend([x_talib_cdl_sum[i],1,1,1,1])
        x.append(vals)
    

    x = np.array(x).astype(float)
    y = np.array(y).astype(float)    

    
    # Remove Some Data (Ancestors) Which could be monotonous
    # like SMA(30) and EMA(20) which dont have good values till index30
    #ancestors = 30    
    #x = x[ancestors:]
    #y = y[ancestors:]    
    '''
    lastXN = len(x)-1
    lastFeature = x[lastXN] # The Enigma Key
    lastFeature = lastFeature.reshape(1,len(lastFeature))
    x = x[:lastXN] # Dropping Last Feature (Key)                    
    '''    
    

    # Considering Only Extremes
    # we have x and y here     
    
    extreme_x = []
    extreme_y = []
    extreme_z = []
    lenNY = len(y)
    for i in range(lenNY):
        if y[i] >= 0 :
            extreme_x.append(x[i])
            extreme_y.append(y[i])
            extreme_z.append(conc_open_closes[i])
            
    x = extreme_x
    y = extreme_y
    z = extreme_z
    
    
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    z = np.array(z).astype(float)
    
    
    lenY = len(y)
    count_y_bulls = 0
    count_y_first = 0
    count_y_scnds = 0
    dbulls = {}
    for i in range(lenY):
        if y[i] == 1 :
            count_y_bulls += 1
            first_bull = z[i][1] - z[i][0]
            second_bull = z[i][2] - z[i][0]
            first_bull = (first_bull*100)/z[i][0]
            second_bull = (second_bull*100)/z[i][0]
            if first_bull > second_bull :
                count_y_first += 1
            else :
                count_y_scnds += 1
            
            sbull = int(second_bull - first_bull)        
                
            if sbull in dbulls.keys() :
                dbulls[sbull] += 1
            else :
                dbulls[sbull] = 1                                            
           
    if count_y_bulls > 0 :
        print('First Bulls    \t',count_y_first/count_y_bulls)
        print('Second Bulls   \t',count_y_scnds/count_y_bulls)
        print('Complete Bulls \t',count_y_bulls)

    dbulls_list_x = []
    dbulls_list_y = []
    dbulls_keys = list(dbulls.keys())
    dbulls_keys.sort()    
    
    for i in dbulls_keys :
        dbulls_list_x.append(i)
        dbulls_list_y.append(dbulls[i])
    
    fig, ax = plt.subplots()                
    ax.scatter(dbulls_list_x, dbulls_list_y)        
    plt.show()
        
    
    ''' Visualize Bullish Sums - Bearish Sums
    Vs Results
    
    
    xsum = np.sum(x, axis = 1)    
    
    xsum_p = np.ma.masked_where(~(y== 1) ,xsum)
    xsum_n = np.ma.masked_where(~(y== 0) ,xsum)
    xsum_e = np.ma.masked_where(~(y==-1) ,xsum)
    
    fig, ax = plt.subplots()            
    ax.plot(xsum,'grey')
    ax.plot(xsum_e,'cyan')
    ax.plot(xsum_p,'green',marker='^')
    ax.plot(xsum_n,'red')
        
    plt.show()
    
    '''
    
    #--------------------------------------------------------------    
    from sklearn.manifold import TSNE    
    
    tsne = TSNE(n_components=2,
                #perplexity=10,
                random_state=1)
                
    
    
    x_clustered = tsne.fit_transform(x)    
        
    x_clustered = np.array(x_clustered).astype(int)
    
    mask_p = np.ma.getmask(np.ma.masked_where(~(y== 1) ,y))
    mask_n = np.ma.getmask(np.ma.masked_where(~(y== 0) ,y))    
    
    mask_p2 = []
    mask_n2 = []
    
    lenMP = len(mask_p)
    for i in range(lenMP):
        mask_p2.append([mask_p[i],mask_p[i]])
        mask_n2.append([mask_n[i],mask_n[i]])
            
    x_clustered_p = np.ma.masked_where(mask_p2, x_clustered)
    x_clustered_n = np.ma.masked_where(mask_n2, x_clustered)    
        
    fig, ax = plt.subplots()                    
    ax.scatter(x_clustered_p[:,0],x_clustered_p[:,1],marker='^',c='green')
    ax.scatter(x_clustered_n[:,0],x_clustered_n[:,1],marker='.',c='red')
        
    plt.show()    
        
    #-------------------------------------------------------------
    
    
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
    prune = False
    while len(pruned_features) < fc :        
    #for ichmoku in range(1,2):
        print( fc-len(pruned_features),'/',fc)
        
        xn = np.delete(x, pruned_features, axis=1)
        fn = np.delete(nfc_names, pruned_features, None)
        
        xtrain = xn[:split_index]        
        xtests = xn[split_index:]
        
        st = datetime.datetime.now()                        
            
        rfc = RandomForestClassifier(n_estimators=est,
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
        prunespeed = 1
        if prune :
            if (fc-cpfc) > prunespeed :
                fkmore = feature_imp.tail(prunespeed).index
                for fkpi in range(prunespeed):                
                    fk = int(fkmore[fkpi])
                    pruned_features.append(fk)
            else :
                break        
        else :
            break
    
    if prune :
        plt.figure(figsize=(20, 16))    
        plt.plot(metro1,label='F1')    
        plt.plot(metro2,label='Recall')
        plt.plot(metro3,label='Accuracy')
        plt.plot(metro4,label='Precision')
        plt.legend()
        plt.grid()
        plt.show()    
            
    






















