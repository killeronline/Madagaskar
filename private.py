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
lenCodes = len(codes_names)
codes = codes_names.keys()
analytics = [['Code','Name','...']]
initTime = datetime.datetime.now()
#codes = ['BOM532480'] # Disaster ALBK
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
    btp_High = 0
    btp_Low = 0
    if bt > 0 :
        bti = len(df)-bt
        bto = df['Open'][bti]
        btc = df['Close'][bti]
        bth = ((df['High'][bti] - df['Open'][bti])*100)/df['Open'][bti]
        btl = ((df['Open'][bti] - df['Low'][bti])*100)/df['Open'][bti]
        btp = ((btc-bto)*100)/bto        
        btp_change = ((int(btp*100))/100)
        bth_change = ((int(bth*100))/100)
        btl_change = ((int(btl*100))/100)
        btm_change = bth_change + btl_change
        df = df[:bti]
        '''            
        if abs(bth_change-btl_change) < 0.5 :
            print('Mixed Stock, Code',code)
            continue
        '''        
            
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
    
    # y = targets
    y = []
    OH = []        
    OL = []    
    chp = 1
    for i in range(m,n-1):                             
        base_z = i + 1
        oprice = yf[opriceColumnName][base_z]        
        hprice = yf[hpriceColumnName][base_z]
        lprice = yf[lpriceColumnName][base_z]        
        ohp = hprice - oprice
        olp = oprice - lprice
        ohp = (ohp*100)/oprice
        olp = (olp*100)/oprice        
        OH.append(ohp)
        OL.append(olp)
        # Remarkable Decision Taken
        change = ohp - olp              
        if change > chp :
            y.append(1) # Signal Buy
        elif change < -chp :
            y.append(0) # Signal Sell 
        else :
            y.append(-1) # Not Effective            
    
    # x = features
    x = np.array(xPrime).astype(float)    
            
    _ ,LenLF = lastFeature.shape
    impactJ = []
    impactStr = []    
    impactCDL = []
    for j in range(LenLF):
        cdl_J = lastFeature[0][j]
        if cdl_J !=0 :
            impactJ.append(j)            
            impactCDL.append(cdl_J)
            impactStr.append(str(j)+'('+str(cdl_J)+')')
    LenJ = len(impactJ)
    if LenJ == 0 :
        print('No Patterns Found, Code',code)
        continue            
        
    '''        
    max_pct = 20
    bins = np.arange(-max_pct, max_pct+0.5) -0.5
    plt.hist(y, bins, alpha=0.5, histtype='bar', ec='black')
    plt.show()
    '''
    # Skimming for more relavant x and y
    z_pos = 0
    z_neg = 0
    z_tot = 0
    xN, xM = x.shape
    z_H = []
    z_L = []   
    OM = []
    for i in range(xN):   
        match = 0
        for j in range(LenJ) :
            if x[i][impactJ[j]] == impactCDL[j] :
                match += 1                
        if match == LenJ : # Ancestor Record
            # Check for Inteference
            noise = 0
            for k in range(xM):
                if k not in impactJ and x[i][k] != 0 :
                    noise += 1                    
                    break
                    
            if noise == 0 :
                if y[i] >= 1 :                                        
                    z_tot += 1
                    z_H.append(OH[i])
                    z_L.append(OL[i])
                    OM.append(OH[i]+OL[i])
                else :
                    continue                
            
    if z_tot == 0 :
        print('No Supportive Evidence, Code',code)
    else :                
        z_ohp = np.array(z_H).astype(float)
        z_olp = np.array(z_L).astype(float)
        z_mag = np.array(OM).astype(float)
        
        z_high_mean = np.mean(z_ohp)
        z_low_mean = np.mean(z_olp)
        z_mag_mean = np.mean(z_mag)
        
        z_high_std = np.std(z_ohp)
        z_low_std = np.std(z_olp)
        z_mag_std = np.std(z_mag)
                
        z_high_mean = int(z_high_mean*100)/100
        z_low_mean = int(z_low_mean*100)/100
        z_mag_mean = int(z_mag_mean*100)/100
        
        z_high_std = int(z_high_std*100)/100
        z_low_std = int(z_low_std*100)/100
        z_mag_std = int(z_mag_std*100)/100

        # Prediction                        
        z_change = z_high_mean - z_low_mean        
        if z_change > 0 : # z_high_mean > z_low_mean
            y_enigma = 1            
            gainLoss = 'Buy'            
        else :
            y_enigma = 0 
            gainLoss = 'Sell'            

        # Cost Analysis        
        costH = 0
        costL = 0        
        if bt > 0 :             
            if y_enigma == 1 :                
                costH = (z_high_mean - z_high_std) - bth_change
                costL = btl_change - (z_low_mean + z_low_std)                
            else :                       
                costH = bth_change - (z_high_mean + z_high_std)
                costL = (z_low_mean - z_low_std) - btl_change
                
            # Clamping High and Low Costs                            
            if costH < 0 :
                costH = 0
                
            if costL < 0 :
                costL = 0
                
            # Cost Function
            cost = costH + costL
                            
        limit = 100000
        zinc = z_tot*LenJ
        # Short Selling Skipped with y_enigma Filter
        if avg_volume > limit and y_enigma == 1 and z_tot > 1 : 
            avg_volume = avg_volume/limit
            avg_volume = int(avg_volume)            
            impactingFts = '_'.join(impactStr)
            potential = (z_high_mean-z_high_std) + (z_low_mean-z_low_std)
            dtS = [last_date_str,code,name,last_close,(avg_volume*int(last_close))]
            dtS += [z_tot,LenJ,zinc,potential,(z_high_std+z_low_std)]
            dtS += [-(z_low_mean-z_low_std),-(z_low_mean+z_low_std)]
            dtS += [z_high_mean,z_high_std,z_low_mean,z_low_std]            
            if bt > 0 :
                dtS += [bth_change,btl_change,cost]                
            analytics.append(dtS)     
            
    if pcc%200 == 0 :
        progress = (pcc*100)//lenCodes
        mtT = 'Progress {} {}_{} ( {}_% )'
        iTime = initTime.strftime('%H_%M')        
        pgText = mtT.format(iTime,pcc,lenCodes,progress)
        mailer.SendEmail(pgText,None)
        
    if pcc > 999 :
        break
    

'''
Sending Results
'''        
header = ['LTDate','Code','Name','LTClose','Capital']
header += ['Zen','LenJ','Zinc','Potential','Risk','RiskBuy','SafeBuy']
header += ['High','HStd','Low','LStd']
if bt > 0 :
    header += ['BTHigh','BTLow','Cost']    
analytics[0] = header


if len(analytics) > 1 :    
    adf = pd.DataFrame(analytics[1:])
    adf.columns = analytics[0]
                
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