# -*- coding: utf-8 -*-
"""
Created on Thu May  9 02:32:24 2019

@author: VAIO
"""

# Load libraries
import os
import sys
import Helpers
import datetime
import pandas as pd

def get_health(filename,percent):    
    try :
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
        
        # Check Last Traded Date
        last_date = df['Date'][0]
        last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()
        today = datetime.datetime.now().date()
        daydifference = (today-last_date).days
        if daydifference > 30 : # Dowload date max cap of dataset files.
            return None
        
        df = df[prices+others]    
        df = df.iloc[::-1]
        df = df.reset_index(drop=True) # Csv Files
        # Cleaning Bad Data
        df = df[df[volumeColumnName]>0]
        df = df.reset_index(drop=True) # Re-Index
        
        # Dropping the 5 Rs Stocks        
        lenDF = len(df)
        avg = sum(df[cpriceColumnName])/lenDF
        if avg < 5 :
            return None
                
        p = 0
        n = 0
        c = 0        
        for i in range(1,df.shape[0]):                        
            cprice = df[cpriceColumnName][i]
            oprice = df[opriceColumnName][i]            
            eff = cprice
            diff =  eff - oprice
            change = (diff*100)/oprice
            if eff > oprice and change > percent :
                p += 1
                c += 1
            elif eff < oprice and change < -percent :
                n += 1
                c += 1
        return [p,n,c]    
    except :
        return None
        
metadata = Helpers.MetaData()
codes = metadata.codes
chn = []
st = datetime.datetime.now()
proc = 0
if len(sys.argv) > 1 :
    proc = int(sys.argv[1]) # One among i%9
    print("Running Evaluation Proc",proc)
i = 0
for (code,name) in codes.items() :    
    i += 1
    if i%9 == proc :
        filename = os.path.join('datasets',code+'.csv')    
        print('Analysing i',i,'code',code)
        if os.path.exists(filename):    
            pi = []
            ni = []
            ci = []
            abort = False
            for percentage in range(1,11):
                health = get_health(filename,percentage)
                if not health :
                    abort = True
                    break
                p,n,c = health
                if c > 0 :
                    pt = p/c
                    nt = n/c
                else :
                    pt = 0
                    nt = 0
                pi.append(pt)
                ni.append(nt)
                ci.append(c)
                
            if not abort : # Aborts include dead stocks and 5 Rs Stocks
                line = str(i)+','+code+','+name+','
                lenPI = len(pi)
                for j in range(lenPI):
                    line += str(pi[j])+','+str(ni[j])+','+str(ci[j])+','            
                chn.append(line+'\n')            
        
et = datetime.datetime.now()        
tt = (et-st).seconds
print("Completed",tt)  
        
contents = ''.join(chn)
f = open('meta\health'+str(proc)+'.csv','w')
f.write(contents)
f.close()

ft = datetime.datetime.now()        
tt = (ft-et).seconds
print("File Written",tt)  


# Extending Merge Functionality
dt = []
for i in range(9):
    f = open('meta\health'+str(i)+'.csv','r')        
    dt.append(f.read())
    f.close()

contents = ''.join(dt)
f = open('meta\healthConsolidated.csv','w')
f.write(contents)
f.close()
