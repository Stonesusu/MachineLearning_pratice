print("you have imported tools")
from .publicLibrary import *
from datetime import datetime,timedelta
class Tools:
    '''
     tools function class
     
     '''
    def __init__(self):
        pass
    
    def string2time(self,x,format='%Y-%m-%d %H:%M:%S'):
        try:
            x = str(x)
            return datetime.strptime(x,format)
        except:
            return np.nan
        
    def time2string(self,x,format='%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strftime(x,format)
        except:
            return np.nan

    def unixtime2time(self,x):
        if x==x:
            x = str(x)
            if len(x)==13:
                x = int(x)//1000
            x = int(x)  
            return datetime.fromtimestamp(x)
        if x!=x:
            return np.nan