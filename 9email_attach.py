import os
import smtplib, ssl
import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText


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
    strDL = ', '.join(distributionList)
    
    body = 'Sent Attachment'         
    subject = 'Kowaski'    
    message = MIMEMultipart()
    message["From"] = froms
    message["To"] = strDL
    message["Subject"] = subject            
    message.attach(MIMEText(body,'plain'))        
    filename = os.path.join('results','atc.csv')
    ctype, encoding = mimetypes.guess_type(filename)
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)
    if maintype == "text":
        fp = open(filename)        
        attachment = MIMEText(fp.read(), _subtype=subtype)
        fp.close()
    elif maintype == "image":
        fp = open(filename, "rb")
        attachment = MIMEImage(fp.read(), _subtype=subtype)
        fp.close()
    elif maintype == "audio":
        fp = open(filename, "rb")
        attachment = MIMEAudio(fp.read(), _subtype=subtype)
        fp.close()
    else:
        fp = open(filename, "rb")
        attachment = MIMEBase(maintype, subtype)
        attachment.set_payload(fp.read())
        fp.close()
        encoders.encode_base64(attachment)
                
    attachment.add_header("Content-Disposition","attachment",filename=filename)
    message.attach(attachment)        
        
    server = smtplib.SMTP("smtp.gmail.com:587")
    server.starttls()
    server.login(froms,passw)
    server.sendmail(froms, distributionList, message.as_string())
    server.quit()
    
else :
    print('Unable to find/load data from',filename)
    



