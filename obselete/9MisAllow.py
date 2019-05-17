import os
import nltk
import Helpers
import pandas as pd

mis_filename = os.path.join('allowed','MisAllowed.csv')
df = pd.read_csv(mis_filename)        
    
metadata = Helpers.MetaData()
codes_names = metadata.codes    
names_codes = {}
for code,name in codes_names.items() :
    names_codes[name] = code

first_names = []
names = []
codes = []
for name,code in names_codes.items() :
    first_names.append(name.split()[0])
    names.append(name)
    codes.append(code)
    
best_matches = ['code,short,name\n']
used_codes = {}
lenDF = len(df)
for i in range(lenDF):    
    print('Processing Number',i+1)
    z = df['code'][i]
    best_diff = 99
    best_word = ''
    best_code = ''
    best_name = ''
    lenFN = len(first_names)
    for fi in range(lenFN) :
        if codes[fi] not in used_codes :            
            f = first_names[fi]
            d = nltk.edit_distance(z,f)
            if d < best_diff and z[0] == f[0] :
                best_diff = d
                best_word = f
                best_code = codes[fi]
                best_name = names[fi]
                used_codes[best_code] = 1
    best_matches.append(','.join([best_code,z,best_name])+'\n')        

contents = ''.join(best_matches)
f = open('allowed/processedAllowed.csv','w')
f.write(contents)
f.close()

