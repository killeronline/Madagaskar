import os
import wget
import quandl
import Helpers

'''
Om Sai Ram

#0 : Get listings of securities
#1 : Get past data till date for all securities
'''

metadata = Helpers.MetaData()
codes = metadata.codes
if not os.path.exists('datasets'):
    os.makedirs('datasets')
urls = []
limit = 30
li = 0
for (code,name) in codes.items() :    
    key = "dt4RSh7B_4EvdsMXnuD2"
    quandl.ApiConfig.api_key = key
    url = "https://www.quandl.com/api/v3/datasets/BSE/"+code+".csv?api_key="+key    
    #downloaded_filename = wget.download(url)
    urls.append(url)
    li += 1
    if li > limit :
        break
    
# Downloading Section Below   
    
    
import sys
import os
import urllib
import threading
from Queue import Queue

class DownloadThread(threading.Thread):
    def __init__(self, queue, destfolder):
        super(DownloadThread, self).__init__()
        self.queue = queue
        self.destfolder = destfolder
        self.daemon = True

    def run(self):
        while True:
            url = self.queue.get()
            try:
                self.download_url(url)
            except Exception,e:
                print "   Error: %s"%e
            self.queue.task_done()

    def download_url(self, url):
        # change it to a different way if you require
        name = url.split('/')[-1]
        dest = os.path.join(self.destfolder, name)
        print "[%s] Downloading %s -> %s"%(self.ident, url, dest)
        urllib.urlretrieve(url, dest)

def download(urls, destfolder, numthreads=4):
    queue = Queue()
    for url in urls:
        queue.put(url)

    for i in range(numthreads):
        t = DownloadThread(queue, destfolder)
        t.start()

    queue.join()

if __name__ == "__main__":
    download(sys.argv[1:], "/tmp")
    
    
    
    
    
    
