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

alltokens = ['code,tradingsymbol,name\n']
for symbol in symbols.keys():
    token = symbols[symbol][0]
    name = symbols[symbol][1]
    alltokens.append(token+','+symbol+','+name+'\n')    
        
contents = ''.join(alltokens)
f = open('allowed/AllTokenCodes.csv','w')
f.write(contents)
f.close()        
        
    
    
    
    




    