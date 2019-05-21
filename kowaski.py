# Load libraries
import os
import csv                                                    # analysis:ignore
import cfg                                                    # analysis:ignore
import sys                                                    # analysis:ignore
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
'''
LD_LIBRARY_PATH = 'LD_LIBRARY_PATH'
hardpath = '/home/sathishphanikurella/Kowaski/LinuxTalib/cTalib/lib'
os.environ[LD_LIBRARY_PATH] = hardpath
'''
    
# After updating paths
import talib

m = 4
mz = 2
bt = 0

pcc = 0
est = 1000 # can be moved to 1000 to check increase in accuracy
split = 50 # can be modified to increase the train and test cases

mailer = Mailers.MailClient()
metadata = Helpers.MetaData()
codes_names = metadata.healthy_codes

codes = codes_names.keys()
lenCodes = len(codes_names)
initTime = datetime.datetime.now()
header1 = ['Code','Name','Samples','LenXN','Threshold']
header2 = ['Strength','Chp','LTClose','LTDate']
analytics = [header1 + header2]
for code in codes :
#def analysis(code,m,mz,chp,est,split,bt):
    pcc += 1        
    verbose = True
    name = codes_names[code]
    st = datetime.datetime.now()
    database_path=os.path.join('database','main.db')
    conn=sql.connect(database_path)
    try :    
        df = pd.read_sql_query('select * from '+code, conn)            
        conn.close()
    except :
        print('Table Not Found:',code)
        conn.close()
        continue
        
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
    n = df.shape[0]    
    if n < 5 : # Not Many Samples
        print('New Stocks1,Code:',code)
        continue
    
    btp = 0
    btp_change = 0
    if bt > 0 :
        bti = len(df)-bt
        bto = df['Open'][bti]
        btc = df['Close'][bti]
        btp = ((btc-bto)*100)/bto        
        btp_change = ((int(btp*100))/100)
        if abs(btp_change) < 0.5 :
            print('Mixed Stock, Code',code)
            continue
        df = df[:bti]
            
    df = df[df[volumeColumnName]>0]
    df = df.reset_index(drop=True)
    n = df.shape[0]    
    if n < 5 : # Not Many Samples
        print('New Stocks2,Code:',code)
        continue
    
    last_date = df['Date'][n-1]
    last_open = df['Open'][n-1]
    last_close= df['Close'][n-1]
    last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()    
    last_date_str = last_date.strftime("%d %b %Y")              

    # improvement : removing dead stocks
    today = datetime.datetime.now().date()
    daydifference = (today-last_date).days
    if daydifference > 10 : # Dead Stock
        print('Old Dead Stock,Code:',code)
        continue            
    
    # Most Required Data    
    df = df[prices+others]    
    yf = df.copy()
        
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
        print('Done Talib, Code',code,' Pcc:',pcc,'/',lenCodes)
    else :
        print('Error At New Talib')

    # Sum of Talib Candles
    x_talib_cdl_sum = np.sum(df, axis = 1)        
    
    # Combining Feature Spaces    
    xPrime = []
    for i in range(m,n):      
        vals = []                
        vals.extend(df.iloc[i].values)        
        vals.extend([x_talib_cdl_sum[i]])
        xPrime.append(vals)
    
    xPrime = np.array(xPrime).astype(float)   
    lastXN = len(xPrime)-1
    lastFeature = xPrime[lastXN] # The Enigma Key
    lastFeature = lastFeature.reshape(1,len(lastFeature))    
    calculatedLenY = n-mz+1-m # Difference of Indices
    xPrime = xPrime[:calculatedLenY] # Dropping Trailing Features    
    
    for chp in range(2,10):
        # Target Variables
        y = []    
        yvolumes = []
        for i in range(m,n-mz+1):        
            mz_diff = []        
            mz_vols = []
            for mzi in range(1,mz):            
                base_z = i + mzi
                oprice = yf[opriceColumnName][base_z]
                cprice = yf[cpriceColumnName][base_z]
                mz_vols.append(yf[volumeColumnName][base_z])
                mz_diff.append(((cprice-oprice)*100)/oprice)                
            avg_diff = sum(mz_diff)/mz
            avg_vols = sum(mz_vols)/mz
            change = avg_diff                
            yvolumes.append(avg_vols)
            if change > chp : # Extreme Positives
                y.append(1)
            elif change < (-chp) : # Extreme Negatives
                y.append(0)
            else :
                y.append(-1) # Mixed Data Point                                    
                              
        y = np.array(y).astype(float)            
        # Considering Only Extremes
        # we have x and y here     
        ygain = 0
        yloss = 0
        yvols = 0
        samples = 0
        extreme_x = []
        extreme_y = []    
        lenNY = len(y)
        for i in range(lenNY):
            if y[i] >= 0 :
                extreme_x.append(xPrime[i])
                extreme_y.append(y[i])
                samples += 1
                yvols += yvolumes[i]
                if y[i] == 1 :
                    ygain += 1
                else :
                    yloss += 1
                        
        # Less Train Data
        if samples < 100 : 
            break # Chp Capped
            
        avg_volume = yvols/samples
        threshold = (abs(ygain-yloss)*100)/samples        
                
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
                
        pcts = np.array(pcts).astype(float)
        totals = np.array(totals).astype(float)
        normDem = max(totals)-min(totals)+1
        occurs = (totals*100)/normDem
        pctsW2 = (2*pcts*occurs)/(pcts+occurs)  #F1=2PR/P+R
        
        best_xEf = ''
        best_rbf = ''
        best_lXN = 0
        best_cur = 0        
        y_enigma = -1        
        ft_pass = 111        
        while ft_pass >= 70 :            
            ft_pass -= 1
            quality_features = []
            for j in range(fc):
                if pcts[j] >= ft_pass :
                    quality_features.append(j)
                        
            xE_Rbf = ''
            x_pass_sum = 0
            robust_features = []
            for j in quality_features :
                x_pass_sum += lastFeature[0][j]
                if lastFeature[0][j] != 0 :
                    robust_features.append(j)                    
                    xE_Rbf += str(j)+'_'
                    
            if x_pass_sum > 0 :
                xh = []
                yh = []                
                lenX = len(x)
                for ki in range(lenX):
                    xh_sum = 0
                    for kj in robust_features :
                        xh_sum += x[ki][kj]
                    if xh_sum > 0 :
                        xh.append([xh_sum,1])
                        yh.append(y[ki])
                        
                xn = np.array(xh).astype(float)
                yn = np.array(yh).astype(float)
                
                # Random Forest
                lenXN = len(xn)                
                split_index = (lenXN * split)//100
                ytrain = yn[:split_index]                
                ytests = yn[split_index:]                   
                xtrain = xn[:split_index]
                xtests = xn[split_index:]                
                if len(xtrain) == 0 or len(xtests) == 0 :                    
                    continue # Not Enough Effective Samples
                        
                clf = RandomForestClassifier(n_estimators=est,
                                     class_weight='balanced',
                                     criterion='gini',
                                     random_state=1,
                                     verbose=False,
                                     n_jobs=-1)
                
                clf.fit(xtrain,ytrain)
                y_pred=clf.predict(xtests)
                cur = metrics.accuracy_score(ytests, y_pred)
                if ( cur > 0.7 and cur > best_cur ):
                    best_cur = cur
                    best_lXN = lenXN
                    xE = np.array([[x_pass_sum,1]]).astype(float)
                    # Enhance 1 = Remove and See Difference
                    clf.fit(xn,yn) # Fitting All Samples
                    y_enigma_pred = clf.predict(xE)
                    y_enigma = int(y_enigma_pred[0])
                    best_xEf = xE_Rbf
                    best_rbf = ''
                    for rbj in robust_features :
                        best_rbf += str(rbj)+'_' 
                
        if y_enigma >=0 :
            if bt > 0 :
                if (y_enigma == 1 and btp > 0) or (y_enigma == 0 and btp < 0) :
                    success = 'Yes'
                else :
                    success = 'No'                
            else :
                success = ''
                btp_change = ''
            
            # Result Filters
            filter1 = (y_enigma == 1 and avg_volume > 100000 and best_lXN > 3)
            filter2 = (samples > 100 and threshold < 30)
            if ( filter1 and filter2 ):
                best_cur = int(best_cur*100)/100                
                dt1 = [code,name,samples,best_lXN,int(threshold)]
                dt2 = [best_cur*100,chp,last_close,last_date_str]
                analytics.append(dt1+dt2)
                break
                
    if pcc%200 == 0 :
        progress = (pcc*100)//lenCodes
        mtT = 'Progress {} {}_{} ( {}_% )'
        iTime = initTime.strftime('%H_%M')        
        pgText = mtT.format(iTime,pcc,lenCodes,progress)
        mailer.SendEmail(pgText,None)    
            
    '''
    if pcc > 20 :
        break    
    '''
    
    
if not os.path.exists('results'):
    os.makedirs('results')

finishdatetime = datetime.datetime.now()
resultfilename = 'R'+finishdatetime.strftime('%d_%b_%Y_%H_%M_%S_%f')+'.csv'
resultfilepath = os.path.join('results',resultfilename)    
with open(resultfilepath,'w+',newline='') as csv_file:
    csvWriter = csv.writer(csv_file,delimiter=',')
    csvWriter.writerows(analytics)

# Send Results
mailer.SendEmail(resultfilename,resultfilename)
    
 
    
    
    
    
    
    



