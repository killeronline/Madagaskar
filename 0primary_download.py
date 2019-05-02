import os
import sys
import wget
import Helpers
import datetime
#import threading
'''
Om Sai Ram

#0 : Get listings of securities
#1 : Download past data for all symbols into "datasets" folder (4 hours)
#2 : Proceed to selective download ( from meta\healths.csv which enlist top 50)
 
'''
def download(code,destfilepath,authkey):    
    url = "https://www.quandl.com/api/v3/datasets/BSE/"+code+".csv?api_key="+authkey            
    wget.download(url,destfilepath)                        
    
metadata = Helpers.MetaData()
codes = metadata.codes.keys()

if not os.path.exists('datasets'):
    os.makedirs('datasets')

authkeys = ["dt4RSh7B_4EvdsMXnuD2",
            "9anNRt9Wdvb59LPKdBEF",
            "jLx3nTvNdTDDgKqU8S9c"]
proc = 0
if len(sys.argv) > 1 :
    proc = int(sys.argv[1]) # One among i%3
i = 0
print("Running proc",proc+1)
st = datetime.datetime.now()
for code in codes :      
    i += 1
    try :                
        if i%3 == proc :
            filepath = os.path.join('datasets',code+'.csv')
            if not os.path.exists(filepath):
                akey = authkeys[i%3]                    
                print(str(i)+" Triggered:"+code)
                download(code,filepath,akey)                        
            #else :
                #print("File Exists:"+code)            
                
                # download(code,filepath,authkeys[t%3])
            # download(code,filepath,akey)
            # import threading        
                #t = threading.Thread(target=download,args=(code,filepath,akey))
                #t.start()                    
            # threading is causing 429 error, too many requests, might get ban                                        
    except :
        print("Error:"+code)

et = datetime.datetime.now()        
tt = (et-st).seconds
print("Completed",tt)        
    


    

    
    
    
