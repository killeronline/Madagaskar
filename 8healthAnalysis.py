# Load libraries
import os
import cfg
import Helpers
import pandas as pd


# use -1 to skip fixPct
fixPct = cfg.zen.chp
thresPct = fixPct-1
threshold = 50
samples = 500
volumethres = 1000 # Minimum 50K Volumes Per Day = Active
filename = os.path.join('meta','healthConsolidated.csv')
df = pd.read_csv(filename,header=None)
lenDF = len(df)
healths = ['code,name,percentage,size,volume\n']
metadata = Helpers.MetaData()
codes = metadata.codes
for i in range(lenDF):
    row = df.iloc[i]    
    code = row[1]
    name = row[2]
    best_p = 0
    best_c = 0
    best_d = 99
    best_v = 0
    for j in range(3,40,4):# 3 to 30
        c = row[j+2]
        if c > samples :
            p = int(row[j+0]*100)
            n = int(row[j+1]*100)        
            v = row[j+3]
            d = abs(p-n)
            pct = j//3
            if d < threshold and d < best_d and pct > thresPct :# Good Pct
                if v > volumethres : # This is must for more active trade
                    if (fixPct < 0) or (pct == fixPct) :
                        best_d = d
                        best_c = c
                        best_p = pct
                        best_v = v
                        break
        else :
            break    
    if best_d < threshold : # Healthy
        line = code+','+name+','+str(best_p)+','+str(best_c)+','
        line += str(best_v)+'\n'
        healths.append(line)
        
lenH = len(healths)
lenRestrict = min(30,len(healths))
print('Healthy Codes :',lenH)
contents = ''.join(healths[0:lenRestrict])# This Measure is only Temporary
f = open('meta\healths.csv','w')
f.write(contents)
f.close()
        
        
        