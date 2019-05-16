# Back Test by deleting last inserted bhav copies
import os
import Helpers
import sqlite3 as sql

metadata = Helpers.MetaData()
codes = metadata.healthy_codes.keys()
numericcodes = []
for code in codes:
    ncode = code[3:]
    numericcodes.append(ncode)    
    
database_path=os.path.join('database','main.db')
conn=sql.connect(database_path)
for ncode in numericcodes:      
    table = 'BOM'+ncode
    cursor=conn.cursor()        
    query = 'delete from '+table+' where id = (select max(id) from '+table+')'
    cursor.execute(query)
    conn.commit()
    
conn.close()