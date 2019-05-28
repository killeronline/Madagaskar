# Set TimeZone
# Trigger Bhav Update     ( Every 3 Hour )
# Run Kowaski Estimator   ( Every 6 Hours )

import os
import time
import datetime

os.environ['TZ'] = 'Asia/Kolkata'


instant = True
elapsed_hours = 0
while True :
    elapsed_hours += 1    
    hour = int(time.strftime('%H'))
    
    st = datetime.datetime.now()   
    
    # Rico
    if instant or hour % 3 == 0 :
        try :
            os.system('python hello.py')
        except :
            print('Failure : Hello')
    
    # Run Bhav
    if instant or hour % 3 == 0 :
        try :
            os.system('python bhavfetch.py')
        except :
            print('Failure : Bhav')
            
    '''
    # Run Kowaski
    if instant or hour % 6 == 0 :
        try :
            os.system('python kowaski.py')
        except :
            print('Failure : Kowaski')
    '''
    # Run BackTest
    if instant or hour == 21 : # 9 PM
        try :
            os.system('python private.py 1') # BackTest
        except :
            print('Failure : BackTest')
            
    # Run Private
    if instant or hour == 6 : # 6 AM
        try :
            os.system('python private.py') # Forecast
        except :
            print('Failure : Private')
        
    et = datetime.datetime.now()    
    tt = (et-st).seconds    
    timetaken = str(tt//60)+' Mins '+str(tt%60)+' Seconds'
    
    # Stats
    print('Elapsed Hours           \t',elapsed_hours)
    print('Last Run Time           \t',datetime.datetime.now())
    print('Time Taken To Compute   \t',timetaken)
    
    if instant :
        print('Manual Run Complete.')
        instant = False
    
    # Sleep    
    rem_minutes = 60 - int(time.strftime('%M'))            
    print('Sleeping ',rem_minutes,'Minutes')
    rem_seconds = (rem_minutes+1)*60
    time.sleep(rem_seconds) # 1 Hour
    