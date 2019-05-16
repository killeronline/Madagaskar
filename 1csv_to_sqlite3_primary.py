import os
import Helpers
import pandas as pd
import sqlite3 as sql


if not os.path.exists('database'):
    os.makedirs('database')

volumeColumnName = 'No. of Shares'
opriceColumnName = 'Open'
hpriceColumnName = 'High'
lpriceColumnName = 'Low'
cpriceColumnName = 'Close'    
ddatesColumnName = 'Date'
ddates = [ddatesColumnName]
prices = [opriceColumnName,
          hpriceColumnName,
          lpriceColumnName,
          cpriceColumnName]
others = [volumeColumnName]
select = ddates + prices + others

database_path=os.path.join('database','main.db')
conn=sql.connect(database_path)


metadata = Helpers.MetaData()
codes = metadata.healthy_codes.keys()
   
i = 0
for code in codes:
    i += 1
    print('Reading',code)
    try :
        csv_file_path = os.path.join('datasets',code+'.csv')
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)       
            df = df[select]
            df = df.iloc[::-1] # Reverse
            df = df.reset_index(drop=True) #Re-Index
            df.to_sql(name=code,
                      con=conn,
                      if_exists='replace',
                      index=True,
                      index_label='id')    
    except :
        print('Failed',code)
        
conn.close()