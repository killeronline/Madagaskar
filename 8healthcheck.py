# Load libraries
import os
import sys
import Helpers
import datetime
import pandas as pd

def get_health(filename):    
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
    except :
        return [0,1]
        
metadata = Helpers.MetaData()
codes = metadata.codes
chn = []
st = datetime.datetime.now()
proc = 0
if len(sys.argv) > 1 :
    proc = int(sys.argv[1]) # One among i%5
    print("Running Evaluation Proc",proc)
i = 0
for (code,name) in codes.items() :    
    i += 1
    if i%5 == proc :        
        filename = os.path.join('datasets',code+'.csv')    
        if os.path.exists(filename):
            if i%100 == 0 :
                print(i,"Evaluating",code)
            p,c = get_health(filename)
            if c > 0 :
                pt = p/c
            else :
                pt = 0
            chn.append(str(i)+","+code+","+name+","+str(pt)+","+str(c)+"\n")      
        
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



        
        