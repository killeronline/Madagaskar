# Load libraries
import os
import pandas as pd

# use -1 to skip fixPct
fixPct = 6
thresPct = 5
threshold = 7
samples = 1000
volumethres = 50000 
filename = os.path.join('meta','healthConsolidated.csv')
df = pd.read_csv(filename,header=None)
lenDF = len(df)
healths = ['code,name,percentage,size,volume\n']
for i in range(lenDF):
    row = df.iloc[i]
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
            if d < threshold and d < best_d and pct > thresPct : # Good Pct
                if v > volumethres : # This is must for more active trade
                    if (fixPct < 0) or (pct == fixPct) :
                        best_d = d
                        best_c = c
                        best_p = pct
                        best_v = v
        else :
            break    
    if best_d < threshold : # Healthy
        line = row[1]+','+row[2]+','+str(best_p)+','+str(best_c)+','
        line += str(best_v)+'\n'
        healths.append(line)
        
contents = ''.join(healths)
f = open('meta\healthPro.csv','w')
f.write(contents)
f.close()
        
        
        