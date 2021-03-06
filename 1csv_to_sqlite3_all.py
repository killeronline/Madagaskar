import os
import Helpers
import pandas as pd
import sqlite3 as sql


if not os.path.exists('database'):
    os.makedirs('database')

metadata = Helpers.MetaData()
codes = metadata.codes.keys()

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
if os.path.exists(database_path):
    os.remove(database_path)
    
# Create Database
conn=sql.connect(database_path)
   
total = len(codes)
i = 0
for code in codes:
    i += 1
    print('Running ',i,'/',total)
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
conn.close()