import os
import wget
import Helpers
'''
Om Sai Ram

#0 : Get listings of securities
#1 : Download past data for all symbols into "datasets" folder (4 hours)
#2 : Process data into sqlite3 database
#3 : Fill missing data from bhav copy
#4 : To do

 
'''
def download(code,destfilepath):
    authkey = "dt4RSh7B_4EvdsMXnuD2"    
    url = "https://www.quandl.com/api/v3/datasets/BSE/"+code+".csv?api_key="+authkey        
    wget.download(url,destfilepath)                        
    
metadata = Helpers.MetaData()
codes = metadata.codes
if not os.path.exists('datasets'):
    os.makedirs('datasets')
for (code,name) in codes.items() :      
    try :
        filepath = os.path.join('datasets',code+'.csv')
        if not os.path.exists(filepath):
            download(code,filepath)
            # import threading
            # threading.Thread(target=download,args=(code,filepath)).start()        
            # threading is causing 429 error, too many requests, might get ban            
        else :
            print("File Exists:"+code)
    except :
        print("Error:"+code)
    


    

    
    
    
