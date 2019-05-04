import smtplib, ssl
port = 465
password = "*"
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    sender = "qq"
    server.login(sender, password)
    server.sendmail(sender, "sa", "Hola\nAmigos")    