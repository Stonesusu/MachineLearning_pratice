print("you have imported tools")
from .publicLibrary import *
from datetime import datetime,timedelta
from dateutil import tz, zoneinfo
import re

class Tools:
    '''
     tools function class
     
     '''
    def __init__(self):
        pass
    
    #当转换失败时，可以用pd.to_datetime，报错信息可以帮助排查异常数据
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

    #生成北京时间
    def unixtime2time(self,x):
        try:
            if x==x:
                x = str(x)
                if len(x)==13:
                    x = int(x)//1000
                else:
                    x = x[:10]
                x = float(x)  
                return datetime.fromtimestamp(x)
            if x!=x:
                return np.nan
        except:
            return np.nan
     
    
    def replaceTime(dt,month=0,days=0,fixedDay=0):
        from calendar import monthrange
        dt=dt.date()
        dt_year = dt.year
        dt_month = dt.month
        dt_day = dt.day
        mon_add = dt_month + month
        if mon_add > 12:
            dt_year = dt_year + int(mon_add / 12)
            dt_month = mon_add % 12
        else:
            dt_month = mon_add
        #直接用会报错replace,比如2020-03-31到2020-04-31
        #获取最终月的总天数
        mondays = monthrange(dt_year,dt_month)[1]
        if dt_day > mondays:
            dt = dt.replace(day=mondays)
        dt = dt.replace(year=dt_year,month=dt_month)
        #fixedDay > 0，则直接设置固定day
        if int(fixedDay) > 0:
            if dt_day > mondays:
                dt = dt.replace(day=mondays)
            else:
                dt = dt.replace(day=int(fixedDay))
        else:
            dt = dt + timedelta(days=days)
        return dt
    

#特殊字符处理
special_character="[-|&|@|(|)|+|!|_|:|\n|']"
def special_characterClean(content):
    # 剔除特殊字符
    content=re.sub(special_character,"",str(content))
    # 去除空格
    content=content.replace(' ','')
    #转小写
    content=content.lower()
    return content