# Load libraries
import os
import pandas as pd
import Helpers

metadata = Helpers.MetaData()
# use -1 to skip fixPct
fixPct = metadata.chp
thresPct = fixPct-1
threshold = 10
samples = 1000
volumethres = 1000 # Minimum 50K Volumes Per Day = Active
filename = os.path.join('meta','healthConsolidated.csv')
df = pd.read_csv(filename,header=None)
lenDF = len(df)
healths = ['code,name,percentage,size,volume\n']
mis_codes = metadata.codes # Mis Allowed Codes Only
for i in range(lenDF):
    row = df.iloc[i]
    if row[1] in mis_codes :
        mis_code = row[1]
        mis_name = mis_codes[mis_code] # Well Formatted Name
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
                if d < threshold and d <= best_d and pct > thresPct :# Good Pct
                    if v > volumethres : # This is must for more active trade
                        if (fixPct < 0) or (pct == fixPct) :
                            best_d = d
                            best_c = c
                            best_p = pct
                            best_v = v
            else :
                break    
        if best_d < threshold : # Healthy
            line = mis_code+','+mis_name+','+str(best_p)+','+str(best_c)+','
            line += str(best_v)+'\n'
            healths.append(line)
        
print('Healthy Codes :',len(healths))        
contents = ''.join(healths)
f = open('meta\healths.csv','w')
f.write(contents)
f.close()
        
        
        