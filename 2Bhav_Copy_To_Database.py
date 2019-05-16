import os
import wget
import Helpers
import datetime
import pandas as pd
import sqlite3 as sql
from zipfile import ZipFile

past = 30
delta = datetime.timedelta(days=1)
i_date = datetime.datetime.now()
i_date -= datetime.timedelta(days=past+1)
dates = []
for i in range(past+1):# including today and early morning bhavcopy (fail safe)
    i_date += delta
    y = i_date.year
    m = i_date.month
    d = i_date.day
    datestr = [] # DD MM YYYY
    if d < 10 :
        datestr.append('0'+str(d))
    else :
        datestr.append(str(d))
    if m < 10 :
        datestr.append('0'+str(m))
    else :
        datestr.append(str(m))
    datestr.append(str(y-2000))    
    datevalue = ''.join(datestr)            
    dates.append(datevalue)
    
#https://www.bseindia.com/download/BhavCopy/Equity/EQ_ISINCODE_260419.zip    
if not os.path.exists('bhav'):
    os.makedirs('bhav')
baseurl = 'https://www.bseindia.com/download/BhavCopy/Equity/'
for date in dates:
    file_name = 'EQ_ISINCODE_'+date
    bhav_csv_file_path = os.path.join('bhav',file_name+'.csv')
    bhav_zip_file_path = os.path.join('bhav',file_name+'.zip')
    if not os.path.exists(bhav_csv_file_path):
        try:
            url = baseurl+file_name+'.zip'
            #print(url)
            print('Fetching bhav copy for',date)
            wget.download(url,bhav_zip_file_path)                        
            with ZipFile(bhav_zip_file_path, 'r') as zip:
                zip.extractall('bhav')
        except:
            pass
            #print('No Bhav Copy for',date)
                    
            
metadata = Helpers.MetaData()
codes = metadata.healthy_codes.keys()
numericcodes = []
ncode_data = {}
for code in codes:
    ncode = code[3:]
    numericcodes.append(ncode)
    ncode_data[ncode] = []
    
tables = {}
for date in dates:    
    file_name = 'EQ_ISINCODE_'+date
    bhav_csv_file_path = os.path.join('bhav',file_name+'.csv')
    if os.path.exists(bhav_csv_file_path):
        st = datetime.datetime.now()
        df = pd.read_csv(bhav_csv_file_path)
        df = df[df['SC_CODE'].isin(numericcodes)]
        df = df.reset_index(drop=True) # Must for array like access
        for i in range(df.shape[0]):            
            scode = str(df['SC_CODE'][i])
            ncode_bhav_arr = []
            ncode_bhav_arr.append(date)
            ncode_bhav_arr.append(df['OPEN'][i])
            ncode_bhav_arr.append(df['HIGH'][i])
            ncode_bhav_arr.append(df['LOW'][i])
            ncode_bhav_arr.append(df['CLOSE'][i])
            ncode_bhav_arr.append(df['NO_OF_SHRS'][i])
            ncode_data[scode].append(ncode_bhav_arr)            
        et = datetime.datetime.now()
        tt = (et-st).seconds
        print('Processed csv file with code',date,'in',tt,'seconds')        


# Judgement Day
database_path=os.path.join('database','main.db')
conn=sql.connect(database_path)
for ncode in numericcodes:      
    table = 'BOM'+ncode
    cursor=conn.cursor()    
    query = 'select id,Date from '+table+' order by id desc limit 1'
    cursor.execute(query)
    last_row = cursor.fetchone()
    last_ider = last_row[0]
    last_date = last_row[1]
    last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()
    #print(ncode,last_date)
    fill_rows = []
    eff_id = int(last_ider) # effective_id
    for arr in ncode_data[ncode]:        
        trade_date = datetime.datetime.strptime(arr[0], "%d%m%y").date()
        #print(trade_date)
        if trade_date > last_date :
            new_bhav_row = []
            dddate = trade_date.strftime("%Y-%m-%d")                        
            oprice = arr[1]
            hprice = arr[2]
            lprice = arr[3]
            cprice = arr[4]
            volume = float(arr[5]) # REAL type in sqlite db
            eff_id += 1 # Auto Increment Primary Key
            row_tuple = (eff_id,dddate,oprice,hprice,lprice,cprice,volume)
            fill_rows.append(row_tuple)
            
    nfill = len(fill_rows)
    if nfill > 0:
        bulk_insert_query = 'insert into '+table+' values (?,?,?,?,?,?,?)'
        conn.executemany(bulk_insert_query,fill_rows) 
        conn.commit()
        print("Committed table",ncode,"Updated",nfill,"rows")     
        
conn.close()  


#tdt = trade_date.strftime("%Y-%m-%d")  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
