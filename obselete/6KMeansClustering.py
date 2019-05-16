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

# BOM530163 - KERALA AYURVEDA LTD. EOD Prices

#def analysis(code,m,chp,est):
proceed = True
code = 'BOM530163'
m = 4
chp = 6
qoff = 0

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
    # Cleaning Bad Data
    df = df[df[volumeColumnName]>0]
    df = df.reset_index(drop=True) #Re-Index
    
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
    cbalance = sum(y)/len(y)
        
    # Include feature(n) : for Last Sample (which is utmost needed)
    past_prices_list = []
    for i in range(m,n+1):
        past_prices_i = []
        base_i = i-m
        for j in range(m):            
            past_prices_i.append(df[lpriceColumnName][base_i+j])
        past_prices_list.append(past_prices_i)        
        
    # Feature Space 1
    x_price_cmp = []            
    for past_prices_i in past_prices_list :    
        cmp_prices_i = []
        cs = len(past_prices_i)
        for ci in range(0,cs-1) :
            for cj in range(ci+1,cs):                  
                den = past_prices_i[ci]
                num = past_prices_i[cj]
                dif = num - den
                pct = dif/num
                cmp_prices_i.append(pct)
                '''
                if past_prices_i[ci] < past_prices_i[cj] :
                    cmp_prices_i.append(1)
                elif past_prices_i[ci] == past_prices_i[cj] :
                    cmp_prices_i.append(5)                    
                else :
                    cmp_prices_i.append(9)                             
                '''
        x_price_cmp.append(cmp_prices_i)
        
    # Feature Space : Percentage
    '''
    x_price_percentages = []            
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
        '''
    
        
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
        
    # Power Drive 1    
    x_macd_line = ta.macd(df[cpriceColumnName],
                          n_fast=12,
                          n_slow=26,
                          fillna = True)
    x_signal = ta.macd_signal(df[cpriceColumnName],
                              n_fast=12,
                              n_slow=26,
                              n_sign=9,
                              fillna = True)
    x_diff = ta.macd_diff(df[cpriceColumnName],
                          n_fast=12,
                          n_slow=26,
                          n_sign=9,
                          fillna = True)
    x_macd = []
    offset = 0
    for i in range(m,n+1):
        #offset += 0.002
        macd_a = 0
        macd_b = 0
        macd_c = 0        
        if i < n :
            macd_a = x_macd_line[i-1]
            macd_b = x_signal[i-1]
            macd_c = x_diff[i-1]
            '''
            t1 = x_diff[i-1]
            t2 = x_diff[i-2]
            if t1 < 0 :
                if t1 < t2 :
                    macd_c = 0
                else :
                    macd_c = 1
            else :
                if t1 < t2 :
                    macd_c = 2
                else :
                    macd_c = 3
            '''
        x_macd.append([macd_a+offset,macd_b+offset,macd_c+offset])
        

    # RSI
    rsi_oversold_level = 30   
    x_rsi = ta.rsi(df[cpriceColumnName],n=14,fillna=True)    
    rsi_overbought_level = 100-rsi_oversold_level
    x_rsi_cmp = [] 
    r_offset = 0
    for i in range(m,n+1):
        r_offset += 0.001
        t_rsi = []
        base_i = i-m
        t_sum = 0
        t_count = 0
        for j in range(m):
            t_rsa = x_rsi[base_i+j]                            
            t_sum += t_rsa
            t_count += 1  
            t_rsi.append((t_rsa/10)+r_offset)
        
        t_avg = t_sum/t_count
        if t_avg > rsi_overbought_level :
            t_k = 5
        elif t_avg > rsi_oversold_level :
            t_k = 3
        else :
            t_k = 1            
        
        x_rsi_cmp.append(t_rsi)
                
        
    # Feature Space 3 : Answer Features
    x_hints = []                
    for i in range(m,n+1):
        if (i == n):
            hz = 0
        else :
            cprice = df[cpriceColumnName][i]
            oprice = df[opriceColumnName][i]                
            eff = cprice
            diff =  eff - oprice
            change = (diff*100)/oprice
            hz = 0
            if eff > oprice and change > chp :            
                hz = 1            
        x_hints.append([hz,hz])
    
    #graph offset
    qq = [ qoff if xh[0] == 1 else 0 for xh in x_hints ]
    
    # Combining Feature Spaces
    x = []
    lenXP = len(x_price_cmp[0])
    lenXM = len(x_macd[0])
    lenXR = len(x_rsi_cmp[0])
    for i in range(m,n+1):
        base_i = i-m
        feature_i = []        
        for xi in range(lenXP):
            x_price_cmp[base_i][xi] += qq[base_i]
        for xj in range(lenXM):
            x_macd[base_i][xj] += qq[base_i]
        for xk in range(lenXR):
            x_rsi_cmp[base_i][xk] += qq[base_i]
        feature_i.extend(x_price_cmp[base_i])
        #feature_i.extend(x_talib_stats[base_i])
        #feature_i.extend(x_macd[base_i])
        #feature_i.extend(x_rsi_cmp[base_i])
        feature_i.extend(x_hints[base_i])        
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



    #--------------------------------------------------------------------------
    
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    xs = ss.fit_transform(x)
    xs_df = pd.DataFrame(xs)
    from pandas.plotting import parallel_coordinates
    lastColumn = xs_df.shape[1]-1
    pc = parallel_coordinates(xs_df,lastColumn,color=('#FF0000', '#00FF00'))
    x = x[:,:-2]
    
    
    #--------------------------------------------------------------------------    
    
    # GRID SEARCH    
    # we have x and y here 
    print('Started Grids')    
    st = datetime.datetime.now()
    len_data = len(x)
    train_percentage = 50
    split_index = (len_data * train_percentage)//100
    xtrain = x[:split_index]
    ytrain = y[:split_index]
    xtests = x[split_index:]
    ytests = y[split_index:]        
    cbalance_train = sum(ytrain)/len(ytrain)
    cbalance_tests = sum(ytests)/len(ytests)
    
    print("cbalance           \t",cbalance)
    print("cbalance train     \t",cbalance_train)
    print("cbalance tests     \t",cbalance_tests)
    
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
    
    '''
    plt.figure(figsize=(20, 16))    
    plt.plot(metro1,label='F1')    
    plt.plot(metro2,label='Recall')
    plt.plot(metro3,label='Accuracy')
    plt.plot(metro4,label='Precision')
    plt.legend()
    plt.grid()
    plt.show()
    '''
    
        
    






















