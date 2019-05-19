# Load libraries
import os
import csv                                                    # analysis:ignore
import cfg
import sys                                                    # analysis:ignore
import talib
import Helpers                                                # analysis:ignore
import Mailers                                                # analysis:ignore
import warnings
import datetime
import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn import metrics                                   # analysis:ignore
import matplotlib.pyplot as plt                               # analysis:ignore
from sklearn.ensemble import RandomForestClassifier           # analysis:ignore

warnings.filterwarnings("ignore")    

#%matplotlib qt

m = 4
mz = 2
bt = 1
est = 1000 # can be moved to 1000 to check increase in accuracy
split = 80 # can be modified to increase the train and test cases
chp = cfg.zen.chp

metadata = Helpers.MetaData()
codes_names = metadata.healthy_codes
codes_nirlon = ['BOM500307'] #NIRLON
#if code is not None :
pcc = 0
analytics = [['Code','Success','Strength','Prediction','Change']]
for code in codes_names.keys() :
#for code in codes_nirlon :
#def analysis(code,m,mz,chp,est,split,bt):
    pcc += 1
    verbose = True
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
    btp = 0
    if bt > 0 :
        bti = len(df)-bt
        bto = df['Open'][bti]
        btc = df['Close'][bti]
        btp = ((btc-bto)*100)/bto        
        df = df[:bti]
            
    df = df[df[volumeColumnName]>0]
    df = df.reset_index(drop=True)
    n = df.shape[0]    
    last_date = df['Date'][n-1]
    last_open = df['Open'][n-1]
    last_close= df['Close'][n-1]
    last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()    
    last_date_str = last_date.strftime("%d %b %Y")              

    # Most Required Data
    df = df[prices+others]

    # Target Variables
    y = []
    avg_diffs = []
    init_opens = []
    conc_diff = []    
    for i in range(m,n-mz+1):
        
        mz_diff = []        
        for mzi in range(1,mz):            
            base_z = i + mzi
            oprice = df[opriceColumnName][base_z]
            cprice = df[cpriceColumnName][base_z]
            mz_diff.append(((cprice-oprice)*100)/oprice)
                        
        conc_diff.append(mz_diff)                
        avg_diff = sum(mz_diff)/mz            
        avg_diffs.append(avg_diff)
        init_opens.append(oprice)
        change = avg_diff                
        if change > chp : # Extreme Positives
            y.append(1)
        elif change < (-chp) : # Extreme Negatives
            y.append(0)
        else :
            y.append(-1) # Mixed Data Point                        
        
    # Feature Space New Talib        
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
        print('Done Talib, Code',code)
    else :
        print('Error At New Talib')

    # Sum of Talib Candles
    x_talib_cdl_sum = np.sum(df, axis = 1)
        
    
    # Combining Feature Spaces    
    x = []
    for i in range(m,n):      
        vals = []                
        vals.extend(df.iloc[i].values)        
        vals.extend([x_talib_cdl_sum[i]])
        x.append(vals)
    
    x = np.array(x).astype(float)    
    y = np.array(y).astype(float)
    
 
    # improvement : stop processing if feature is zero vector

    # Enigma
    #hbcp = cp[n-1]
    #hbop = op[n-1]
    lastXN = len(x)-1
    lastFeature = x[lastXN] # The Enigma Key
    lastFeature = lastFeature.reshape(1,len(lastFeature))    
    x = x[:len(y)] # Dropping Trailing Features

    # Considering Only Extremes
    # we have x and y here     
    extreme_x = []
    extreme_y = []    
    lenNY = len(y)
    for i in range(lenNY):
        if y[i] >= 0 :
            extreme_x.append(x[i])
            extreme_y.append(y[i])            
            
    x = extreme_x
    y = extreme_y            
    
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)  
    
    lenX = len(x)
    fc = len(x[0])
    positives = [0]*fc
    negatives = [0]*fc
    totals = [0]*fc    
    for i in range(lenX):
        for j in range(fc):
            cdl = x[i][j] 
            if cdl != 0 :
                if  (cdl > 0 and y[i] == 1) or (cdl < 0 and y[i] == 0):
                    positives[j] += 1
                else :
                    negatives[j] += 1
                    
                totals[j] += 1            

    pcts = [0]*fc                
    for j in range(fc):
        if (totals[j] > 0):
            pcts[j] = (positives[j]/totals[j])*100
    
    '''        
    fig, ax = plt.subplots()                    
    ax.plot(positives,marker='^',c='green')    
    ax.plot(negatives,marker='*',c='red')    
    ax.plot(totals,marker='.',c='blue')    
    ax.plot(pcts,marker='.',c='cyan')    
    plt.show()
    '''

    ft_pass = 100
    while ft_pass >= 70 :
        robust_features = []    
        for j in range(fc):
            if pcts[j] >= ft_pass :
                robust_features.append(j)
        x_pass_sum = 0
        for j in robust_features :
            x_pass_sum += lastFeature[0][j]
            
        if x_pass_sum != 0 :
            break
        else :
            ft_pass -= 1
    
    x_refined = []    
    for i in range(lenX):
        ft_refined = []
        for j in robust_features :            
            ft_refined.append(x[i][j])                
        x_refined.append(ft_refined)
    
    x = x_refined
    x = np.array(x).astype(float)

    # Enigma Prediction
    x_enigma_sum = 0
    for j in robust_features :
        x_enigma_sum += lastFeature[0][j]
        
    if x_enigma_sum == 0 :
        y_enigma = -1   # No Cdl
    else :
        if x_enigma_sum > 0 :
            y_enigma = 1    # Bull
        else :
            y_enigma = 0    # Bear
    
    success = '?'
    if y_enigma == -1 :
        success = '?'
    else :
        if (y_enigma == 1 and btp > 0) or (y_enigma == 0 and btp < 0) :
            success = 'Yes'
        else :
            success = 'No'
                
        analytics.append([code,success,ft_pass,y_enigma,btp])    
    
    
if not os.path.exists('results'):
    os.makedirs('results')

finishdatetime = datetime.datetime.now()
resultfilename = 'A'+finishdatetime.strftime('%d_%b_%Y_%H_%M_%S_%f')+'.csv'
resultfilepath = os.path.join('results',resultfilename)    
with open(resultfilepath,'w+',newline='') as csv_file:
    csvWriter = csv.writer(csv_file,delimiter=',')
    csvWriter.writerows(analytics)

    
mailer = Mailers.MailClient()
mailer.SendEmail(resultfilename,resultfilename)
    
 
    
    
    
    
    
    



