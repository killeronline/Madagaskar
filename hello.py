import Mailers
import datetime

initTime = datetime.datetime.now()
iTime = initTime.strftime('%H_%M')
pgText = 'Rico, {}'.format(iTime)


mailer = Mailers.MailClient()
mailer.SendEmail(pgText,None)