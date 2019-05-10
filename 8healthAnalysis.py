# Load libraries
import os
import pandas as pd

# use -1 to skip fixPct
fixPct = 6
thresPct = 5
threshold = 5
samples = 1000
filename = os.path.join('meta','healthConsolidated.csv')
df = pd.read_csv(filename,header=None)
lenDF = len(df)
healths = []
for i in range(lenDF):
    row = df.iloc[i]
    best_p = 0
    best_c = 0
    best_d = 99
    for j in range(3,31,3):# 3 to 30
        c = row[j+2]
        if c > samples :
            p = int(row[j+0]*100)
            n = int(row[j+1]*100)        
            d = abs(p-n)
            pct = j//3
            if d < threshold and d < best_d and pct > thresPct : # Good Pct
                if (fixPct < 0) or (pct == fixPct) :
                    best_d = d
                    best_c = c
                    best_p = pct
        else :
            break    
    if best_d < threshold : # Healthy
        line = row[1]+','+row[2]+','+str(best_p)+','+str(best_c)+'\n'
        healths.append(line)
        
contents = ''.join(healths)
f = open('meta\healthPro.csv','w')
f.write(contents)
f.close()
        
        
        