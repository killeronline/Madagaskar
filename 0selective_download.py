import os
import wget
import Helpers
import datetime
#import threading
'''
Om Sai Ram

#2 : Selective download from healths.csv

 
'''
def download(code,destfilepath,authkey):    
    url = "https://www.quandl.com/api/v3/datasets/BSE/"+code+".csv?api_key="+authkey            
    wget.download(url,destfilepath)                        
    
metadata = Helpers.MetaData()
codes = metadata.healthy_codes.keys()

if not os.path.exists('datasets'):
    os.makedirs('datasets')

authkeys = ["dt4RSh7B_4EvdsMXnuD2",
            "9anNRt9Wdvb59LPKdBEF",
            "jLx3nTvNdTDDgKqU8S9c"]

st = datetime.datetime.now()
i = 0
for code in codes :      
    i += 1
    #try :                
    filepath = os.path.join('datasets',code+'.csv')
    if not os.path.exists(filepath):
        akey = authkeys[0]                    
        print(str(i)+" Triggered:"+code)
        download(code,filepath,akey)                        
    #except :
        #print("Error:"+code)

et = datetime.datetime.now()        
tt = (et-st).seconds
print("Completed",tt)        
    


    

    
    
    
