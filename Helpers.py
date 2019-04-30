import os
import pandas as pd

class MetaData():

    def get_codes_dict(self):
        code_filename = os.path.join('meta','codes.csv')
        df = pd.read_csv(code_filename)        
        codes = {}
        eod = 0 
        for i in range(df.shape[0]):
            if df['code'][i][:3] == 'BOM' :
                eod += 1
                codes[df['code'][i]] = df['name'][i]        
            else :
                break        
        return codes
    
    def __init__(self):
        self.codes = self.get_codes_dict()
        
    