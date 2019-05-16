# Load libraries
import os
import sys
import Helpers
import datetime
import pandas as pd

def get_health(i,code,name,filename,percent):    
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
        if daydifference > 30 : # Download date max cap of dataset files.
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

        line = str(i)+','+code+','+name+','                     
        for pct in range(1,percent):
            p = 0
            n = 0 
            c = 0
            v = 0
            for i in range(1,df.shape[0]):                        
                cprice = df[cpriceColumnName][i]
                oprice = df[opriceColumnName][i]            
                volume = df[volumeColumnName][i]
                eff = cprice
                diff =  eff - oprice
                change = (diff*100)/oprice
                if eff > oprice and change > pct :
                    p += 1
                    c += 1
                    v += volume
                elif eff < oprice and change < -pct :
                    n += 1
                    c += 1
                    v += volume
                    
            if c > 0 :
                p = p/c
                n = n/c
                v = v//c                
                line += str(p)+','+str(n)+','+str(c)+','+str(v)+','
            else :
                line += str(p)+','+str(n)+','+str(c)+','+str(v)+','
                
        return line + '\n'
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
            health = get_health(i,code,name,filename,11)
            if health :                    
                chn.append(health)
        
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

