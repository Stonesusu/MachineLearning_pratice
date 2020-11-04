import pandas as pd
import numpy as np
def getBins(data,column,bin_num):
    '''
    分箱函数
    data: 原始数据集
    column 需要分箱的列
    bin_num 分箱数
    '''
    try:
        rp_data,bins = pd.qcut(list(data[column]),bin_num,duplicates='drop',retbins=True)
        if len(list(bins))>2:
            bins=[-np.inf]+list(bins[1:-1])+[np.inf]
            right=True
        else:
            start=bins[0]
            end=bins[1]
            temp=data[(data[column]>start)&(data[column]<end)]
            if temp.shape[0]>0:
                rp_data,bins = pd.qcut(list(temp[column]),bin_num-2,duplicates='drop',retbins=True)
                bins=[-np.inf]+[start]+list(bins)+[end]+[np.inf]
                right=True
            else:
                bins=[-np.inf]+list(bins)+[np.inf]
                right=True
    except:
        rp_data,bins = pd.qcut(list(data[column]),1,duplicates='drop',retbins=True)
        bins=[-np.inf]+[list(bins)[0]]+[np.inf]
        right=False
    bins=sorted(set(bins))
    return bins,right
def getWoe(A,A_sum,B,B_sum,defaults=1):
    '''
    除数为0情况处理
    '''
    A_pct=A/A_sum
    B_pct=B/B_sum
    if B_pct==0:
        woe=defaults
    else:
        if A_pct==0:
            woe=-defaults
        else:
            woe=np.log(A_pct/B_pct)
    return woe

def weightProcessing(dataset,weight=None,target='target',good_event=1):
    '''
    好坏权重: 默认 [好,坏]=[1,1]
    如果好坏权重不一样,请带上权重列,并指定列名
    '''
    data=dataset
    if weight is None:
        data['weight']=data.apply(lambda x: 1,axis=1)
    elif type(weight)==list:
        if len(weight)!=2:
            data['weight']=data.apply(lambda x: 1,axis=1)
        else: 
            data['weight']=data[target].apply(lambda x: weight[0] if x==good_event else weight[1])
    elif type(weight)==str:
        if weight!='weight':
            data=data.rename(columns={weight:'weight'})
    data=data.reset_index(drop=True)
    return data

def getIvWoe(result,column,good_sum,bad_sum):
    temp=pd.DataFrame(columns=[column,'bad','good','badRate','Total','good_pct','bad_pct','Odds','WOE','IV'])
    if result.shape[0]>0:
        temp=result.groupby(column)['bad','good'].sum().reset_index()
        temp=temp[(temp['bad']>0)|(temp['good']>0)]
        if temp.shape[0]>0:
            temp['badRate']=temp.apply(lambda x: x.bad/(x.bad+x.good),axis=1)
            temp['Total']=temp.apply(lambda x: x.bad+x.good,axis=1)
            temp['good_pct']=temp['good'].apply(lambda x: x/good_sum)
            temp['bad_pct']=temp['bad'].apply(lambda x:x/bad_sum)
            temp['Total_pct']=temp['Total'].apply(lambda x: x/(good_sum+bad_sum))
            temp['Odds']=temp.apply(lambda x: x.good/x.bad if x.bad>0 else np.nan,axis=1)
            temp['WOE']=temp.apply(lambda x: getWoe(x.good,good_sum,x.bad,bad_sum),axis=1)
            temp['IV']=temp.apply(lambda x:(x.good/good_sum-x.bad/bad_sum)*x.WOE,axis=1)
            temp=temp.replace(np.inf,np.nan).replace(-np.inf,np.nan)
    return temp



def scoreBoxSingle(dataset,cut=10,score='score',target='target',weight=None,good_event=1):
    
    '''
    评分分箱 函数
    需要 [评分:score,标签:target,权重:weight] 字段
    good_event target 中 好的值   
    '''
    
    def process(df):
        df['badrate']=df.apply(lambda x: x.bad/(x.good+x.bad) if x.good+x.bad>0 else np.nan,axis=1)
        return df
    data=weightProcessing(dataset,weight=weight,target=target,good_event=good_event) 
    max_score=data[score].max()
    min_score=data[score].min()
    x=range((int(min_score)//cut+1)*cut,(int(max_score)//cut+1)*cut,cut)
    bins=[-np.inf]+list(x)+[np.inf]
    cuts=pd.DataFrame(pd.cut(data[score],bins=bins,right=False))
    cuts.columns=['cuts']
    data=pd.concat([data,cuts],axis=1)
    data['good']=data.apply(lambda x: x.weight if x[target]==good_event else 0,axis=1)
    data['bad']=data.apply(lambda x: x.weight if x[target]!=good_event else 0,axis=1)
    good_sum=data['good'].sum()
    bad_sum=data['bad'].sum()
    temp=data.groupby(['cuts'])['bad','good'].sum().reset_index()
    for col in ['bad','good']:
        temp[col]=temp[col].fillna(0)
    temp=process(temp)
    temp['population_pct']=temp.apply(lambda x:(x['good']+x['bad'])/(good_sum+bad_sum),axis=1)
    temp['cumsum_pct']=temp['population_pct'].cumsum()
    return temp   

def scoreBox(dataset,cut=10,score='score',flag='flag',train_flag='mdl',valid_flag='vld',target='target',weight=None,good_event=1):
        '''
        dataset 数据集(含训练集和验证集)
        score  评分
        cut    评分分段
        flag   训练集和验证集的字段
        train_flag flg 中训练集值
        valid_flag flg 中验证集值
        target 好坏标签字段
        weight 样本权重
        good_event target 中 好的值   
        ''' 
        #权重处理
        data=weightProcessing(dataset,weight=weight,target=target,good_event=good_event) 
        
        max_score=data[score].max()
        min_score=data[score].min()
        x=range((int(min_score)//cut+1)*cut,(int(max_score)//cut+1)*cut,cut)
        bins=[-np.inf]+list(x)+[np.inf]
        cuts=pd.DataFrame(pd.cut(data[score],bins=bins,right=False))
        cuts.columns=['cuts']
        data=pd.concat([data,cuts],axis=1)
        
        data['good']=data.apply(lambda x: x.weight if x[target]==good_event else 0,axis=1)
        data['bad']=data.apply(lambda x: x.weight if x[target]!=good_event else 0,axis=1)
        
        temp=data.groupby(['cuts',flag])['bad','good'].sum().reset_index()
        mdl=temp[temp[flag]==train_flag][['cuts','good','bad']]
        vld=temp[temp[flag]==valid_flag][['cuts','good','bad']]
        for col in ['bad','good']:
            mdl[col]=mdl[col].fillna(0)
            vld[col]=vld[col].fillna(0)
        def process(df):
            df['badrate']=df.apply(lambda x: x.bad/(x.good+x.bad) if x.good+x.bad>0 else np.nan,axis=1)
            return df
        mdl=process(mdl)
        vld=process(vld)
        return mdl,vld
    
def getScoreBins(data,column,cut=10,dynamicCut=True):
    max_score=data[column].max()
    min_score=max(0,data[column].min())
    x=range((int(min_score)//cut+1)*cut,(int(max_score)//cut+1)*cut,cut)
    if dynamicCut:
        while len(x)<5:
            cut= int(cut/2)
            x=range((int(min_score)//cut+1)*cut,(int(max_score)//cut+1)*cut,cut)
            if cut<2:
                break
        while len(x)>20:
            cut=cut*2
            x=range((int(min_score)//cut+1)*cut,(int(max_score)//cut+1)*cut,cut)
    bins=[0]+list(x)+[np.inf]
    return bins,False

def getWeightBins(dataset,cut=10,score='score',target='target',weight='weight',good_event=1): 
    def process(df):
        df['badrate']=df.apply(lambda x: x.bad/(x.good+x.bad) if x.good+x.bad>0 else np.nan,axis=1)
        return df
    data=weightProcessing(dataset,weight=weight,target=target,good_event=good_event) 
    weight_sum=data['weight'].sum()
    data=data.sort_values(score,ascending=True).reset_index(drop=True)
    data['cumsum']=data['weight'].cumsum()
    weight_bins=[-np.inf]+[weight_sum/cut*(i+1) for i in range(cut-1)]+[np.inf]
    bins=set()
    for index,weight_bin in enumerate(weight_bins[:-2]):
        bins.add(data[(data['cumsum']>=weight_bin)&(data['cumsum']<weight_bins[index+1])][score].max())
    bins=list(bins)
    bins=sorted(bins)
    bins=[-np.inf]+bins+[np.inf]
    cuts=pd.DataFrame(pd.cut(data[score],bins=bins,right=False))
    cuts.columns=['cuts']
    data=pd.concat([data,cuts],axis=1)    
    data['good']=data.apply(lambda x: x.weight if x[target]==good_event else 0,axis=1)
    data['bad']=data.apply(lambda x: x.weight if x[target]!=good_event else 0,axis=1)
    good_sum=data['good'].sum()
    bad_sum=data['bad'].sum()
    temp=data.groupby(['cuts'])['bad','good'].sum().reset_index()
    for col in ['bad','good']:
        temp[col]=temp[col].fillna(0)
    temp=process(temp)
    temp['population_pct']=temp.apply(lambda x:(x['good']+x['bad'])/(good_sum+bad_sum),axis=1)
    temp['cumsum_pct']=temp['population_pct'].cumsum()
    return temp

def badRateSequence(badRate):
    '''
    坏账率是从小到大还是从大到小
    True  从大到小
    False 从小到大
    '''
    all_num=len(badRate)
    result=[]
    for index in range(all_num-1):
        result.append(badRate[index+1]-badRate[index])
    Increment=[x for x in result if x>=0]
    Decrement=[x for x in result if x<0]
    Increment_num=len(Increment)
    Decrement_num=len(Decrement)
    if Decrement_num==all_num-1 or Decrement_num>=Increment_num:
        return True
    return False


def scoreBoxByPct(dataset,cut=10,score='score',target='target',weight='weight',good_event=1,badRateSort=True,is_int=True,process_missing=False): 
    def process(df):
        df['badrate']=df.apply(lambda x: x.bad/(x.good+x.bad) if x.good+x.bad>0 else np.nan,axis=1)
        return df
    df=weightProcessing(dataset,weight=weight,target=target,good_event=good_event) 
    data_missing=df[(df[score]<0)|(df[score].isnull())]
    data=df[df[score]>=0]
    if process_missing:
         weight_sum=df['weight'].sum()
    else:
        weight_sum=data['weight'].sum()
    data=data.sort_values(score,ascending=True).reset_index(drop=True)
    score_temp=score+'_temp'
    if is_int:
        data[score_temp]=data[score].apply(lambda x: int(x+0.5))
    else:
        data[score_temp]=data[score]
    data['cumsum']=data['weight'].cumsum()
    data_normal_weight_sum=data['weight'].sum()
    weight_bins=[-np.inf]+[data_normal_weight_sum/cut*(i+1) for i in range(cut-1)]+[np.inf]
    bins=set()
    for index,weight_bin in enumerate(weight_bins[:-2]):
        bins.add(data[(data['cumsum']>=weight_bin)&(data['cumsum']<weight_bins[index+1])][score_temp].max())
    bins=list(bins)
    bins=sorted(bins)
    bins=[-np.inf]+bins+[np.inf]
    cuts=pd.DataFrame(pd.cut(data[score],bins=bins,right=False))
    cuts.columns=['cuts']
    data=pd.concat([data,cuts],axis=1)
    data['good']=data.apply(lambda x: x.weight if x[target]==good_event else 0,axis=1)
    data['bad']=data.apply(lambda x: x.weight if x[target]!=good_event else 0,axis=1)
    temp=data.groupby(['cuts']).agg({'bad':'sum','good':'sum',score:'median'}).reset_index()
    temp.rename(columns={score:'median'},inplace=True)
    for col in ['bad','good']:
        temp[col]=temp[col].fillna(0)
    temp=process(temp)
    if process_missing:
        if not data_missing.empty:
            bad_sum=data_missing[data_missing[target]!=good_event].weight.sum()
            good_sum=data_missing[data_missing[target]==good_event].weight.sum()
            tmp=pd.DataFrame([['NAN',bad_sum,good_sum]],columns=['cuts','bad','good'])
            tmp=process(tmp)
            tmp['median']=-9999979
            temp=pd.concat([tmp,temp])
            temp=temp[(temp['bad']>0)|(temp['good']>0)]
            temp=temp.reset_index(drop=True)
    if badRateSort:
        if not badRateSequence(list(temp[temp['cuts']!='NAN']['badrate'])):
            temp=temp.sort_values('median',ascending=False).reset_index()
    temp['population_pct']=temp.apply(lambda x:(x['good']+x['bad'])/weight_sum,axis=1)
    temp['cumsum_pct']=temp['population_pct'].cumsum()
    temp=temp[['cuts','bad','good','badrate','population_pct','cumsum_pct','median']]
    return temp



from sklearn.utils.multiclass import type_of_target
import multiprocessing
import sys

class WoeIvToolsWithMultiProgress:
    def __init__(self,process=True,processes=10):
        self.__cpu_count=multiprocessing.cpu_count()
        self.__platform=sys.platform
        if process:
            if self.__platform in ['linux','darwin']:
                self.__process=True
                self.__processes=min(processes,self.__cpu_count)
        else:
            self.__process=False
            self.__processes=processes
                
    def partitionData(self,data,column,processMissing,negativeMissing):
        '''
        processMissing   是否对缺失值单独分箱
        negativeMissing  是否将负值判断为缺失
        '''
        # 正常数值类型特征
        # 类型特征
        if data[column].dtype in [float,int]:
            if processMissing:
                if negativeMissing:
                    missing_data=data[(data[column].isnull())|(data[column]<0)][[column,'good','bad']].reset_index(drop=True)
                    normal_data=data[data[column]>=0][[column,'good','bad']].reset_index(drop=True)
                else:
                    missing_data=data[data[column].isnull()][[column,'good','bad']].reset_index(drop=True)
                    normal_data=data[data[column]==data[column]][[column,'good','bad']].reset_index(drop=True)
                return normal_data,missing_data
            else:
                return data[[column,'good','bad']],pd.DataFrame(columns=[column,'good','bad'])
        else:
            df=data[[column,'good','bad']]
            if processMissing:
                df[column]=data[column].fillna('NAN')
            else:
                df=df[df[column]==df[column]].reset_index(drop=True)
            df[column]=df[column].astype(str)
            return df,pd.DataFrame(columns=[column,'good','bad'])
        
    def numericalWoeMethod(self,data,column,bin_num,processMissing,negativeMissing,good_sum,bad_sum,bins,right):
        normal_data,missing_data=self.partitionData(data,column,processMissing,negativeMissing)
        if normal_data[column].dtype in [float,int]:
            if not normal_data.empty:
                if bins is None:
                    bins,right=getBins(normal_data,column,bin_num)
                    if negativeMissing:
                        if bins[1]!=0 and min(normal_data[column])>0:
                            bins=[0]+bins[1:]
                        else:
                            bins=[-0.001]+bins[1:]
                #正常数据分箱
                normal_cuts=pd.DataFrame(pd.cut(normal_data[column],bins=bins,right=right))
                normal_cuts.columns=[column]
                normal_cuts=pd.concat([normal_cuts,normal_data[['good','bad']]],axis=1)
                resul_normal_data=getIvWoe(normal_cuts,column,good_sum,bad_sum)
            else:
                resul_normal_data=pd.DataFrame(columns=[column,'bad','good','badRate','Total','good_pct','bad_pct','Odds','WOE','IV'])
            if not missing_data.empty:
                missing_data[column]=missing_data[column].apply(lambda x: 'NAN')
                resul_missing_data=getIvWoe(missing_data,column,good_sum,bad_sum)
                resul_data=pd.concat([resul_missing_data,resul_normal_data])  
            else:
                resul_data=resul_normal_data
        else:
            resul_data=getIvWoe(normal_data,column,good_sum,bad_sum)
        resul_data=resul_data.reset_index(drop=True)
        return resul_data,bins,right
    
    def getTrainIV(self,dataset,bin_num=10,target='target',weight='weight',good_event=1,processMissing=True,negativeMissing=True,bins_dict={},right_dict={}):
        '''
        processMissing   是否对缺失值单独分箱
        negativeMissing  是否将负值判断为缺失
        ''' 
        #权重处理
        data=weightProcessing(dataset,weight=weight,target=target,good_event=good_event)
        '''
        样本标签
        样本标签的值必须为 二值型
        '''
        y_type = type_of_target(data[target])
        if y_type not in ['binary']:
            raise ValueError('Label type must be binary')
            
        data['good']=data.apply(lambda x: x.weight if x[target]==good_event else 0,axis=1)
        data['bad']=data.apply(lambda x: x.weight if x[target]!=good_event else 0,axis=1)
        #好样本数量
        good_sum=data['good'].sum()
        #坏样本数量
        bad_sum=data['bad'].sum()
        iv=[]
        woes={}
        if self.__process:
            pool = multiprocessing.Pool(processes=self.__processes)
            executeResults={}
            for column in data.columns:
                if column not in [target,'weight','good','bad']:
                    bins = bins_dict.get(column)
                    right = right_dict.get(column)
                    executeResults[column]=pool.apply_async(func=self.numericalWoeMethod,args=(data,column,bin_num,processMissing,negativeMissing,good_sum,bad_sum,bins,right))
            pool.close()
            pool.join()
            for column,value in executeResults.items():
                resul_data,bins,right=value.get()
                woes[column]=resul_data
                iv.append([column,resul_data['IV'].sum()])
                bins_dict[column]=bins
                right_dict[column]=right
        else:
            for column in data.columns:
                if column not in [target,'weight','good','bad']:
                    bins=bins_dict.get(column)
                    right=right_dict.get(column)
                    resul_data,bins,right=self.numericalWoeMethod(data,column,processMissing,negativeMissing,good_sum,bad_sum,bins,right)
                    iv.append([column,resul_data['IV'].sum()])
                    bins_dict[column]=bins
                    right_dict[column]=right
        iv=pd.DataFrame(iv,columns=['feature','IV'])
        iv=iv.sort_values('IV',ascending=False).reset_index(drop=True)
        return iv,woes,bins_dict,right_dict
    def getIV(self,dataset,bin_num=10,target='target',weight='weight',good_event=1,processMissing=True,negativeMissing=True):
        iv,woes,bins_dict,right_dict=self.getTrainIV(dataset,bin_num,target,weight,good_event,processMissing,negativeMissing,bins_dict={},right_dict={})
        return iv,woes  
    
    def getTrainValidIV(self,train,test,valid,weight=None,bin_num=10,target='target',good_event=1,\
                        consistent=True,processMissing=True,negativeMissing=True):
        '''
        train,test,valid DataFrame,其中变量要做过二元化
        target 标签列
        good_event 好标签对应的值
        consistent 训练集合验证集的变量分箱是否保持一致
        processMissing 是否对缺失值单独处理
        negativeMissing 负值是否划分到缺失值中
        
        ''' 
        if test is not None:
            train_data=pd.concat([train,test])
        else:
            train_data=train.copy()
        valid_data=valid.copy()
        iv_train,train_woes,bins_dict,right_dict = self.getTrainIV(dataset=train_data,bin_num=bin_num,target=target,weight=weight,good_event=good_event,processMissing=processMissing,negativeMissing=negativeMissing,bins_dict={},right_dict={})                                            
        if consistent:
            iv_valid,valid_woes,bins_dict,right_dict =self.getTrainIV(dataset=valid_data,bin_num=bin_num,target=target,weight=weight,good_event=good_event,processMissing=processMissing,negativeMissing=negativeMissing,bins_dict=bins_dict,right_dict=right_dict)                                            
        else:
            iv_valid,valid_woes,bins_dict,right_dict =self.getTrainIV(dataset=valid_data,bin_num=bin_num,target=target,weight=weight,good_event=good_event,processMissing=processMissing,negativeMissing=negativeMissing,bins_dict={},right_dict={})                                            
        iv_train.rename(columns={'IV':'iv_train'},inplace=True)
        iv_valid.rename(columns={'IV':'iv_valid'},inplace=True)
        iv=pd.merge(iv_train,iv_valid,on='feature',how='left')
        iv.sort_values('iv_train',ascending=False).reset_index(drop=True)
        return iv,train_woes,valid_woes
    

class CsiToolsWithMultiProgress:
    def __init__(self,process=True,processes=10):
        self.__cpu_count=multiprocessing.cpu_count()
        self.__platform=sys.platform
        if process:
            if self.__platform in ['linux','darwin']:
                self.__process=True
                self.__processes=min(processes,self.__cpu_count)
        else:
            self.__process=False
            self.__processes=processes
    
    def dataProcess(self,data,column,bin_num,processMissing,negativeMissing):
        '''
        processMissing   是否对缺失值单独分箱
        negativeMissing  是否将负值判断为缺失
        '''
        if data[column].dtype in [float,int]:
            if processMissing:
                if negativeMissing:
                    missing_data=data[(data[column].isnull())|(data[column]<0)][[column,'mdl','vld']].reset_index(drop=True)
                    normal_data=data[data[column]>=0][[column,'mdl','vld']].reset_index(drop=True)
                else:
                    missing_data=data[data[column].isnull()][[column,'mdl','vld']].reset_index(drop=True)
                    normal_data=data[data[column]==data[column]][[column,'mdl','vld']].reset_index(drop=True)
                if not normal_data.empty:
                    bins,right=getBins(normal_data,column,bin_num)
                    if negativeMissing:
                        if bins[1]!=0 and min(normal_data[column])>0:
                            bins=[0]+bins[1:]
                        else:
                            bins=[-0.001]+bins[1:]   
                    normal_cuts=pd.DataFrame(pd.cut(normal_data[column],bins=bins,right=right))
                    normal_cuts.columns=[column]
                    normal_data=pd.concat([normal_cuts,normal_data[['mdl','vld']]],axis=1)
                else:
                    normal_data=pd.DataFrame(columns=[column,'mdl','vld'])

                if not missing_data.empty:
                    missing_data[column]=missing_data[column].apply(lambda x: 'NAN')
                    resul_data=pd.concat([missing_data,normal_data])
                else:
                    resul_data=normal_data
            else:
                bins,right=getBins(data,column,bin_num)
                cuts=pd.DataFrame(pd.cut(data[column],bins=bins,right=right))
                cuts.columns=[column]
                resul_data=pd.concat([cuts,data[['mdl','vld']]],axis=1)

        else:
            resul_data=data[[column,'mdl','vld']]
            if processMissing:
                resul_data[column]=resul_data[column].fillna('NAN')
            else:
                resul_data=resul_data[resul_data[column]==resul_data[column]].reset_index(drop=True)
            resul_data[column]=resul_data[column].astype(str)
        return resul_data
    def calculationPsi(self,data,column,bin_num,processMissing,negativeMissing,train_sum,valid_sum):
        resul_data=self.dataProcess(data,column,bin_num,processMissing,negativeMissing)
        temp=resul_data.groupby(column)['mdl','vld'].sum().reset_index()
        temp['mdl_pct']=temp['mdl'].apply(lambda x: x/train_sum)
        temp['vld_pct']=temp['vld'].apply(lambda x: x/valid_sum)
        temp['csi']=temp.apply(lambda x: getWoe(x.mdl,train_sum,x.vld,valid_sum)*(x.mdl/train_sum-x.vld/valid_sum),axis=1)
        temp=temp[(temp['mdl']>0)|(temp['vld']>0)].reset_index(drop=True)
        return temp
    def getPSI(self,train_data,test_data,valid_data,bin_num=10,weight=None,invalid_ftr=['target'],processMissing=True,negativeMissing=True):
        
        #数据集合并
        if test_data is not None:
            train=pd.concat([train_data,test_data])
        else:
            train=train_data.copy()
        valid=valid_data.copy()
        flag='TrainValidFlag'
        train[flag]='mdl'
        valid[flag]='vld'
        
        #数据集权重处理
        if weight is not None:
            if type(weight)==str:
                train=train.rename(columns={weight:'weight'})
                valid=valid.rename(columns={weight:'weight'})
            else:
                raise '输入样本无法找到权重'
        else:
            train['weight']=train.apply(lambda x: 1,axis=1)
            valid['weight']=valid.apply(lambda x: 1,axis=1)
                 
        train_sum=train['weight'].sum()
        valid_sum=valid['weight'].sum()
        data=pd.concat([train,valid])

        data['mdl']=data.apply(lambda x: x['weight'] if x[flag]=='mdl' else 0,axis=1)
        data['vld']=data.apply(lambda x: x['weight'] if x[flag]=='vld' else 0,axis=1)
        columns=data.columns
        psi=[]
        psi_dict={}
        if self.__process:
            pool = multiprocessing.Pool(processes=self.__processes)
            executeResults={} 
            for column in columns:
                if column not in [flag,'mdl','vld','weight'] and column not in invalid_ftr:
                     executeResults[column]=pool.apply_async(func=self.calculationPsi,args=(data,column,bin_num,processMissing,negativeMissing,train_sum,valid_sum))
            pool.close()
            pool.join()
            for column,value in executeResults.items():
                temp=value.get()
                psi_dict[column]=temp
                psi.append([column,temp['csi'].sum()])
        else:
            for column in columns:
                if column not in [flag,'mdl','vld','weight'] and column not in invalid_ftr:
                    temp=self.calculationPsi(data,column,bin_num,processMissing,negativeMissing,train_sum,valid_sum)
                    psi_dict[column]=temp
                    psi.append([column,temp['csi'].sum()])
        psi=pd.DataFrame(psi,columns=['feature','CSI'])
        psi.sort_values('CSI',ascending=False).reset_index(drop=True)
        return psi,psi_dict
