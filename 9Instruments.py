import os
import pandas as pd

mis_filename = os.path.join('allowed','instruments.csv')
df = pd.read_csv(mis_filename)
df = df[df['exchange']=='BSE']
df = df[df['instrument_type']=='EQ']
df = df[df['name'].astype(str) != 'nan']
df = df.reset_index(drop=True)
df = df[['exchange_token','tradingsymbol','name']]    
symbols = {}
lenDF = len(df)
for i in range(lenDF):
    token = 'BOM'+str(df['exchange_token'][i])
    if token[0:4] == 'BOM5':
        symbol = df['tradingsymbol'][i]
        name = df['name'][i]
        symbols[symbol] = [token,name]

# Allowed Symbols, Tokens, Names
mis_filename = os.path.join('allowed','MisAllowed.csv')
df = pd.read_csv(mis_filename) 
lenDF = len(df)
allowed = ['code,tradingsymbol,name\n']
for i in range(lenDF):
    scrip = df['code'][i]
    if scrip in symbols :
        token = symbols[scrip][0]
        name = symbols[scrip][1]        
        allowed.append(token+','+scrip+','+name+'\n')
        
contents = ''.join(allowed)
f = open('allowed/HealthyMisCodes.csv','w')
f.write(contents)
f.close()        
        
    
    
    
    




    