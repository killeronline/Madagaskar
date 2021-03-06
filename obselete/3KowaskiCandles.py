# Load libraries
import os
import csv                                                    # analysis:ignore
import cfg
import sys                                                    # analysis:ignore
import talib
import Helpers
import Mailers                                                # analysis:ignore
import warnings
import datetime
import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn import metrics
import matplotlib.pyplot as plt                               # analysis:ignore
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")    

#%matplotlib qt
def analysis(code,m,mz,chp,est,split,bt):
    verbose = False
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
        
    if verbose :
        print("==================================================================")            
    # Cleaning Data
    # Remove All Non Traded Days ( Volume = 0 )
    df = df[df[volumeColumnName]>0]
    df = df.reset_index(drop=True)
    n = df.shape[0]    
    last_date = df['Date'][n-1]
    last_open = df['Open'][n-1]
    last_close= df['Close'][n-1]
    last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()    
    last_date_str = last_date.strftime("%d %b %Y")
    
    # some parameter checks 
    if mz < 2 :
        return None
    
    # 5 Rs Stocks
    if last_close < 10 :
        print('Low Cap Stock')
        return None
    
    # improvement : improve healths.csv file to exclude dead stocks        
    today = datetime.datetime.now().date()
    daydifference = (today-last_date).days
    if daydifference > 5 : # Dead Stock
        print('Old Dead Stock')
        return None            

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
        print('Done Talib')
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
    
    
    #print('Started Grids')        
    len_data = len(x)
    train_percentage = split
    split_index = (len_data * train_percentage)//100
    xtrain = x[:split_index]
    ytrain = y[:split_index]
    xtests = x[split_index:]
    ytests = y[split_index:]

    #classbalance_y = sum(y)/len(y)
    #classbalance_ytrain = sum(ytrain)/len(ytrain)
    #classbalance_ytests = sum(ytests)/len(ytests)    
    #print("balance y          \t",classbalance_y)
    #print("balance ytrain     \t",classbalance_ytrain)
    #print("balance ytests     \t",classbalance_ytests)        
    

    fc = len(xtrain[0])    
    # %matplotlib qt    
    
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
    prune = False
    while len(pruned_features) < fc :        
    #for ichmoku in range(1,2):
        #print( fc-len(pruned_features),'/',fc)
        
        xn = np.delete(x, pruned_features, axis=1)        
        
        xtrain = xn[:split_index]        
        xtests = xn[split_index:]        
            
        clf = RandomForestClassifier(n_estimators=est,
                                     class_weight='balanced',
                                     criterion='gini',
                                     random_state=1,
                                     verbose=False,
                                     n_jobs=-1)
        clf.fit(xtrain,ytrain)

        y_pred=clf.predict(xtests)           
        
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
        #mtn, mfp, mfn, mtp = metrics.confusion_matrix(ytests, y_pred).ravel()                                
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
            
            # Making Prediction            
            xfinal = lastFeature                                                    
            yfinal=clf.predict(xfinal)
            yprobs=clf.predict_proba(xfinal)            
            ybull= yprobs[0][1]                   
            
        if not prune :
            break                
    
    y_pred = best_predictions            
    clf = best_clf  # Restoring Best Model                
    '''
    print("M                \t",m)    
    print("MZ               \t",mz)
    print("Chp              \t",chp)    
    print("F1Sc             \t",best_f1score)
    print("Recall           \t",best_recall)
    print("Accuracy         \t",best_acc)
    print("Precision        \t",best_prec)
    print("Estimators       \t",est)
    print("Company Code     \t",code)        
    print("Feature Count    \t",best_feature_count)
    '''
    
    if verbose :
        print("F1Sc             \t",best_f1score)
        print("Recall           \t",best_recall)
        print("Accuracy         \t",best_acc)
        print("Precision        \t",best_prec)
        print("Feature Count    \t",best_feature_count)
    
    et = datetime.datetime.now()
    tt = (et-st).seconds    
    timetaken = str(tt//60)+' Mins '+str(tt%60)+' Seconds'
    
    if verbose :
        print('\n'+timetaken)
    
    # half blood = second bull impact relative to magnitude of first bull
    # op = i open prices
    # cp = i close prices
    '''
    hb = ((hbcp - hbop)*100)/hbop
    hbchp = chp
    if yfinal[0] == 0 :
        hbchp = -chp
    exp = ((mz*hbchp)-hb)/(mz-1)    
    hbp = exp
    reg = last_close+(hbp*last_close/100) # regressed close
    '''
    success = '?'
    if bt > 0 :
        if yfinal[0] == 1 and btp > 0 :
            success = 'Yes' # Buying
        elif yfinal[0] == 0 and btp < 0 :
            success = 'Yes' # Shorting
        else :
            success = 'No'

    result = []    
    used_params = [code,m,mz,chp,est]
    time_metric = [timetaken,tt]
    best_metric = [best_prec,best_acc,best_recall,best_f1score]
    best_output = [last_open,last_close,last_date_str,ybull,yfinal[0]]    
    best_bcktst = [btp,success]
    #best_halfbs = [hbp,reg]
    result.extend(used_params)
    result.extend(time_metric)
    result.extend(best_metric)
    result.extend(best_output)
    result.extend(best_bcktst)
    #result.extend(best_halfbs)    
    y_signal = yfinal[0]
    y_bull_rounded = int(ybull*100)/100
    #best_acc_rounded = int(best_acc*100)/100 
    best_prec_rounded = int(best_prec*100)/100
    
    clean = [last_date_str,last_close,best_prec_rounded,y_bull_rounded,y_signal]
    if bt > 0 :
        return clean + best_bcktst
    else :
        return clean

# Main of Kowaski  
# def analysis(code,m,mz,chp,est,split):        
# Generic Hypers
m = 4
mz = 2
bt = 1
est = 1000 # can be moved to 1000 to check increase in accuracy
split = 80 # can be modified to increase the train and test cases
chp = cfg.zen.chp
headers1 = ['Code','PastDays','FutureDays','ChangeInPercentage','Estimators']
headers2 = ['Total Time Taken','Seconds']
headers3 = ['Precision','Accuracy','Recall','F1Score']
headers4 = ['LastOpen','LastClose','LastDate','BullPower','Prediction']
headers5 = ['ActualChange','Success']
headers6 = ['Code','Name']
#headers' = ['HalfBloodPrince','RegressedClose']
#headers = headers1 + headers2 + headers3 + headers4 + headers5 + headers6
headers = ['Last Traded Date','Last Close','Precision']
headers += ['Whiskey Bulls','Buy/Short']
if bt > 0 :
    headers += ['Actual Change','Success']
    
headers += ['BSE Code','Stock Name']
results = [headers]
metadata = Helpers.MetaData()
codes_names = metadata.healthy_codes
wins = 0
loss = 0
tots = 0
quality_missed = 0
skipped_code_count = 0
iterated_code_count = 0
processed_code_count = 0
for (code,name) in codes_names.items():    
    iterated_code_count += 1
    print('Iterating Code',iterated_code_count)
    result = analysis(code,m,mz,chp,est,split,bt)
    # Accuracy > 60% and Bulls are Top 40% and Bears are Bottom 40%
    if result :
        processed_code_count += 1
        print('Processed Code',code,'index',processed_code_count)                
        #if result[2] > 0.6 and
        if result[2] > 0.6 and (result[3] < 0.4 or result[3] > 0.6) and result[4] > 1 :        
            tots += 1
            if result[6] == 'Yes' :
                wins += 1
            else :
                loss += 1
            
            results.append(result+[code,name])   
            #results.append(result+[code,name])   
            #results.append(result)       
        else :
            quality_missed += 1
            
    else :
        skipped_code_count += 1

#print('Filter',quality_missed)    
#print('Skipped',skipped_code_count)
#print('Iterated',iterated_code_count)
#print('Processed',processed_code_count)    
print('******************************')
print('chp',chp)
if bt > 0 and tots > 0 :
    print('Process        \t',processed_code_count)
    print('Quality        \t',tots)
    print('Success Rate   \t',wins/tots)


if not os.path.exists('results'):
    os.makedirs('results')

finishdatetime = datetime.datetime.now()
resultfilename = finishdatetime.strftime('%d_%b_%Y_%H_%M_%S_%f')+'.csv'
resultfilepath = os.path.join('results',resultfilename)    
with open(resultfilepath,'w+',newline='') as csv_file:
    csvWriter = csv.writer(csv_file,delimiter=',')
    csvWriter.writerows(results)

    
mailer = Mailers.MailClient()
mailer.SendEmail(resultfilename,resultfilename)






















