import os
import Helpers
import datetime
import pandas as pd
import sqlite3 as sql



past = 2
delta = datetime.timedelta(days=1)
i_date = datetime.datetime.now()
i_date -= datetime.timedelta(days=past+1)
for i in range(past+1):# including today and early morning bhavcopy (fail safe)
    i_date += delta
    y = i_date.year
    m = i_date.month
    d = i_date.day
    datestr = [] # DD_MM_YYYY
    if d < 10 :
        datestr.append('0'+str(d))
    else :
        datestr.append(str(d))
    if m < 10 :
        datestr.append('0'+str(m))
    else :
        datestr.append(str(m))
    datestr.append(str(y))    
    datevalue = ''.join(datestr)        
    print(i_date)
    print(datevalue)
