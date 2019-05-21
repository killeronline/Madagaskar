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
bt = 2

pcc = 0
est = 1000 # can be moved to 1000 to check increase in accuracy
split = 50 # can be modified to increase the train and test cases

mailer = Mailers.MailClient()
metadata = Helpers.MetaData()
codes_names = metadata.healthy_codes
lenCodes = len(codes_names)
codes = codes_names.keys()
analytics = [['Code','Name','...']]
initTime = datetime.datetime.now()
#codes = ['BOM512531'] # SCI
#codes = ['BOM532488'] # DIVIS
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
    
    avg_volume = int(sum(df[volumeColumnName])/n)
    
    last_date = df['Date'][n-1]
    last_open = df['Open'][n-1]
    last_close= df['Close'][n-1]
    last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()    
    last_date_str = last_date.strftime("%d %b")              

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
        #vals.extend([x_talib_cdl_sum[i]])
        xPrime.append(vals)
    
    xPrime = np.array(xPrime).astype(float)   
    lastXN = len(xPrime)-1
    lastFeature = xPrime[lastXN] # The Enigma Key
    lastFeature = lastFeature.reshape(1,len(lastFeature))    
    calculatedLenY = n-mz+1-m # Difference of Indices
    xPrime = xPrime[:calculatedLenY] # Dropping Trailing Features  
    
    _ ,LenLF = lastFeature.shape
    impactJ = []
    impactCDL = []
    for j in range(LenLF):
        cdl_J = lastFeature[0][j]
        if cdl_J !=0 :
            impactJ.append(j)
            impactCDL.append(cdl_J)
    LenJ = len(impactJ)
    if LenJ == 0 :
        print('No Patterns Found, Code',code)
        continue
    
    xLee = []
    a,b = xPrime.shape
    for i in range(a):
        xtemp = []        
        for j in impactJ :            
            xtemp.append(xPrime[i][j])                    
        xLee.append(xtemp)
        
    x = np.array(xLee).astype(float)
    
    chp = 1
    
    y = []        
    for i in range(m,n-mz+1):        
        mz_diff = []                
        for mzi in range(1,mz):            
            base_z = i + mzi
            oprice = yf[opriceColumnName][base_z]
            cprice = yf[cpriceColumnName][base_z]            
            mz_diff.append(((cprice-oprice)*100)/oprice)                
        avg_diff = sum(mz_diff)/mz        
        change = avg_diff                        
        #y.append( round(change) )        
        if change > chp :
            y.append(1)
        elif change < -chp :
            y.append(0)  
        else :
            y.append(-1)
        
    '''        
    max_pct = 20
    bins = np.arange(-max_pct, max_pct+0.5) -0.5
    plt.hist(y, bins, alpha=0.5, histtype='bar', ec='black')
    plt.show()
    '''

    z_pos = 0
    z_neg = 0
    z_tot = 0
    xN, xM = x.shape
    for i in range(xN):   
        match = 0
        for j in range(LenJ) :
            if x[i][j] == impactCDL[j] :
                match += 1                
        if match == LenJ :
            if y[i] == 1 :
                z_pos += 1
                z_tot += 1
            elif y[i] == 0 :
                z_neg += 1
                z_tot += 1
            
    if z_tot == 0 :
        print('No Supportive Evidence, Code',code)
    else :
        score = 0
        y_enigma = 0
        strength = z_pos/z_tot        
        gainLoss = 'Sell'
        if strength > 0.5 :
            y_enigma = 1
            score = (2*strength) - 1.0
            gainLoss = 'Buy'
        else :
            y_enigma = 0            
            score = 1.0 - (2*strength)
            
        score = int(score*100.0)/100.0        
        
        if bt > 0 :
            if (y_enigma == 1 and btp > 0) or (y_enigma == 0 and btp < 0) :
                success = 'Pass'
            else :
                success = 'Fail'                
        else :
            success = ''
            btp_change = ''            
                
        limit = 100000
        if avg_volume > limit :                            
            avg_volume = avg_volume/limit
            avg_volume = int(avg_volume)
            zen = str(z_pos)+'_'+str(z_neg)+'_'+str(z_tot)
            dtS = [last_date_str,last_close,avg_volume,code,name]
            dtS += [gainLoss,score,zen,LenJ]
            if bt > 0 :
                dtS += [btp_change,success]
            analytics.append(dtS)     
            
    if pcc%200 == 0 :
        progress = (pcc*100)//lenCodes
        mtT = 'Progress {} {}_{} ( {}_% )'
        iTime = initTime.strftime('%H_%M')        
        pgText = mtT.format(iTime,pcc,lenCodes,progress)
        mailer.SendEmail(pgText,None)
    
    if pcc > 20 :
        break
'''
Sending Results
'''        
header = ['LTDate','LTClose','V','Code','Name','GL','Score','Zen','LenJ']
if bt > 0 :
    header += ['BTChp','BTValidate']
analytics[0] = header
                
if not os.path.exists('results'):
    os.makedirs('results')

finishdatetime = datetime.datetime.now()
resultfilename = 'R'+finishdatetime.strftime('%d_%b_%Y_%H_%M_%S')+'.csv'
resultfilepath = os.path.join('results',resultfilename)    
with open(resultfilepath,'w+',newline='') as csv_file:
    csvWriter = csv.writer(csv_file,delimiter=',')
    csvWriter.writerows(analytics)

# Email Results
mailer.SendEmail(resultfilename,resultfilename)