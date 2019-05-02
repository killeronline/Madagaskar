# Load libraries
import os
import datetime
import pandas as pd

def get_health(filename):        
    df = pd.read_csv(filename)
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
    df = df.iloc[::-1]
    df = df.reset_index(drop=True)
    p = 0
    c = 0
    percent = 1
    for i in range(1,df.shape[0]):            
        c += 1
        cprice = df[cpriceColumnName][i]
        oprice = df[opriceColumnName][i]            
        eff = cprice
        diff =  eff - oprice
        change = (diff*100)/oprice
        if eff > oprice and change > percent :
            p += 1
    return [p,c]        
        
chn = []
st = datetime.datetime.now()
i = 0
codes = [
'BOM531562',
'BOM538652',
'BOM505585',
'BOM512014',
'BOM512309',
'BOM536993',
'BOM507952',
'BOM531628',
'BOM532068',
'BOM537867',
'BOM539304',
'BOM501368',
'BOM512063',
'BOM537648'
        ]
for code in codes :
    filename = os.path.join('datasets',code+'.csv')    
    if os.path.exists(filename):    
        p,c = get_health(filename)
        if c > 0 :
            pt = p/c
        else :
            pt = 0
        chn.append(code+","+str(pt)+","+str(c)+"\n")      
        
et = datetime.datetime.now()        
tt = (et-st).seconds
print("Completed",tt)  
