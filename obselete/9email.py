import os
import smtplib, ssl

def message(text):
    return 'Subject: {}\n\n{}'.format('Kowaski', text)

try :
    filename = os.path.join('meta','zemail.txt')
    file = open(filename,'r')
    data = file.readlines()
    file.close()
    
    lenData = len(data)        
    if lenData > 0 :
        port = 465
        passw = data[0]
        froms = data[1]
        distributionList=[]
        for i in range(2,lenData):
            distributionList.append(data[i])
            
        context = ssl.create_default_context()    
        with smtplib.SMTP_SSL('smtp.gmail.com', port, context=context) as server:
            server.login(froms, passw)
            server.sendmail(froms, distributionList, message('Greetings'))    
            
    else :
        print('Unable to find/load data from',filename)
except Exception as e:
    print('Exception:',str(e))