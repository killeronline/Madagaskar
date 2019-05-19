import os
import pandas as pd

class MetaData():

    def get_codes_dict(self):
        code_filename = os.path.join('meta','codes.csv')
        if os.path.exists(code_filename):
            df = pd.read_csv(code_filename)        
            codes = {}        
            for i in range(df.shape[0]):
                if df['code'][i][:3] == 'BOM' :                
                    codes[df['code'][i]] = df['name'][i]        
                else :
                    break        
            return codes
        else :
            return None
    
    def get_healthy_dict(self):
        code_filename = os.path.join('meta','healths.csv')
        if os.path.exists(code_filename):
            df = pd.read_csv(code_filename)        
            codes = {}
            for i in range(df.shape[0]):
                codes[df['code'][i]] = df['name'][i]
            return codes
        else :
            return None
    
    def __init__(self):
        self.codes = self.get_codes_dict()
        self.healthy_codes = self.get_healthy_dict()        
        
    