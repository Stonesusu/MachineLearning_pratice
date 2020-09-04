print("you have imported featureEngineering")

from .publicLibrary import *
import multiprocessing

#变量相关性分析
def fea_corr_analysis(df,target,exclusion_cols,corr_threshold):
    corr = df.drop(exclusion_cols,axis=1).corr()
    corr_del = []
    for col in corr.columns:
        if col!=target:
            tmp = abs(corr[col])
            tmp = list(tmp[tmp>corr_threshold].index)
            if len(tmp)<1:
                continue
            if col in tmp:
                tmp.remove(col)
            if target in tmp:
                tmp.remove(target)
            for i in tmp:
                #该特征与label的相关性大于col与label的相关性，因此remove
                if corr.loc[target,i]>corr.loc[target,col]:
                    tmp.remove(i)
            corr_del.extend(tmp)
        else: continue
    corr_del = list(set(corr_del))
    print('count of train corr del:{}'.format(len(corr_del)))
    return corr_del

#直接填充缺失值,并生成新的变量
def fill_null_directly(df,cols,fill_value=-99999):
    for col in cols:
        df[col+'_null'] = df[col].replace(np.nan,fill_value)
    return df

#按缺失值：1，非缺失：0处理；
def fill_is_null(df,cols):
    for ele in cols:
        df[ele+'_isnull'] = np.where(df[ele].isnull(),1,0)
    return df

def fill_missingvalue(train,test=None,exclusion_cols=[],mode='mean'):
    def get_mean_median_mode(df,mode):
        if mode=='mean':
            return df.mean()
        elif mode=='median':
            return df.median()
        elif mode=='mode': #比较耗时
            return df.mode().T[0]  
        else:
            return np.nan
    train = train.drop(exclusion_cols,axis=1)
    fill_value = get_mean_median_mode(train,mode)
    train = train.fillna(fill_value)
    if test is not None:
        test = test.fillna(fill_value)
    return train,test

# 定义一个卡方分箱（可设置参数置信度水平与箱的个数）停止条件为大于置信水平且小于bin的数目
def ChiMerge(self,df, variable, flag, confidenceVal=3.841, bin=10, sample = None):  
    '''
    运行前需要 import pandas as pd 和 import numpy as np
    df:传入一个数据框仅包含一个需要卡方分箱的变量与正负样本标识（正样本为1，负样本为0）
    variable:需要卡方分箱的变量名称（字符串）
    flag：正负样本标识的名称（字符串）
    confidenceVal：置信度水平（默认是不进行抽样95%）
    bin：最多箱的数目
    sample: 为抽样的数目（默认是不进行抽样），因为如果观测值过多运行会较慢
    '''
    #进行是否抽样操作
    if sample != None:
        df = df.sample(n=sample)
    else:
        df   

    #进行数据格式化录入
    total_num = df.groupby([variable])[flag].count()  # 统计需分箱变量每个值数目
    total_num = pd.DataFrame({'total_num': total_num})  # 创建一个数据框保存之前的结果
    positive_class = df.groupby([variable])[flag].sum()  # 统计需分箱变量每个值正样本数
    positive_class = pd.DataFrame({'positive_class': positive_class})  # 创建一个数据框保存之前的结果
    regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True,
                       how='inner')  # 组合total_num与positive_class
    regroup.reset_index(inplace=True)
    regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']  # 统计需分箱变量每个值负样本数
    regroup = regroup.drop('total_num', axis=1)
    np_regroup = np.array(regroup)  # 把数据框转化为numpy（提高运行效率）
    print('已完成数据读入,正在计算数据初处理')

    #处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
    i = 0
    while (i <= np_regroup.shape[0] - 2):
        if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or ( np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
            np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 正样本
            np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 负样本
            np_regroup[i, 0] = np_regroup[i + 1, 0]
            np_regroup = np.delete(np_regroup, i + 1, 0)
            i = i - 1
        i = i + 1

    #对相邻两个区间进行卡方值计算
    chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值
    for i in np.arange(np_regroup.shape[0] - 1):
        chi = (np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 \
          * (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) / \
          ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * (
          np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
        chi_table = np.append(chi_table, chi)
    print('已完成数据初处理，正在进行卡方分箱核心操作')

    #把卡方值最小的两个区间进行合并（卡方分箱核心）
    while (1):
        if (len(chi_table) <= (bin - 1) and min(chi_table) >= confidenceVal):
            break
        chi_min_index = np.argwhere(chi_table == min(chi_table))[0]  # 找出卡方值最小的位置索引
        np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
        np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
        np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
        np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

        if (chi_min_index == np_regroup.shape[0] - 1):  # 最小值试最后两个区间的时候
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                       ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index, axis=0)

        else:
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                       * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                       ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[chi_min_index, 2]))
            # 计算合并后当前区间与后一个区间的卡方值并替换
            chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                       * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                   ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
    print('已完成卡方分箱核心操作，正在保存结果')

    #把结果保存成一个数据框
    result_data = pd.DataFrame()  # 创建一个保存结果的数据框
    result_data['variable'] = [variable] * np_regroup.shape[0]  # 结果表第一列：变量名
    list_temp = []
    for i in np.arange(np_regroup.shape[0]):
        if i == 0:
            x = '0' + ',' + str(np_regroup[i, 0])
        elif i == np_regroup.shape[0] - 1:
            x = str(np_regroup[i - 1, 0]) + '+'
        else:
            x = str(np_regroup[i - 1, 0]) + ',' + str(np_regroup[i, 0])
        list_temp.append(x)
    result_data['interval'] = list_temp  # 结果表第二列：区间
    result_data['flag_0'] = np_regroup[:, 2]  # 结果表第三列：负样本数目
    result_data['flag_1'] = np_regroup[:, 1]  # 结果表第四列：正样本数目

    return result_data

#CART最优分箱
def get_bestsplit_list(self,sample_set, var,target):
    '''
    根据分箱得到最优分割点list
    param sample_set: 待切分样本
    param var: 分割变量名称
    '''
    def calc_score_median(sample_set, var):
        '''
        计算相邻评分的中位数，以便进行决策树二元切分
        param sample_set: 待切分样本
        param var: 分割变量名称
        '''
        var_list = list(np.unique(sample_set[var]))
        var_median_list = []
        for i in range(len(var_list) -1):
            var_median = (var_list[i] + var_list[i+1]) / 2
            var_median_list.append(var_median)
        return var_median_list

    def bining_data_split(sample_set, var, min_sample, split_list,target):
        '''
        划分数据找到最优分割点list
        param sample_set: 待切分样本
        param var: 分割变量名称
        param min_sample: 待切分样本的最小样本量(限制条件)
        param split_list: 最优分割点list
        '''
        split, position = choose_best_split(sample_set, var, min_sample,target)
        if split != 0.0:
            split_list.append(split)
        # 根据分割点划分数据集，继续进行划分
        sample_set_left = sample_set[sample_set[var] < split]
        sample_set_right = sample_set[sample_set[var] > split]
        # 如果左子树样本量超过2倍最小样本量，且分割点不是第一个分割点，则切分左子树
        if len(sample_set_left) >= min_sample * 2 and position not in [0.0, 1.0]:
            bining_data_split(sample_set_left, var, min_sample, split_list,target)
        else:
            None
        # 如果右子树样本量超过2倍最小样本量，且分割点不是最后一个分割点，则切分右子树
        if len(sample_set_right) >= min_sample * 2 and position not in [0.0, 1.0]:
            bining_data_split(sample_set_right, var, min_sample, split_list,target)
        else:
            None

    def choose_best_split(sample_set, var, min_sample,target):
        '''
        使用CART分类决策树选择最好的样本切分点
        返回切分点
        param sample_set: 待切分样本
        param var: 分割变量名称
        param min_sample: 待切分样本的最小样本量(限制条件)
        '''
        # 根据样本评分计算相邻不同分数的中间值
        score_median_list = calc_score_median(sample_set, var)
        median_len = len(score_median_list)
        sample_cnt = sample_set.shape[0]
        sample1_cnt = sum(sample_set[target])
        sample0_cnt = sample_cnt- sample1_cnt
        Gini = 1 - np.square(sample1_cnt / sample_cnt) - np.square(sample0_cnt / sample_cnt)

        bestGini = 0.0; bestSplit_point = 0.0; bestSplit_position = 0.0
        for i in range(median_len):
            left = sample_set[sample_set[var] < score_median_list[i]]
            right = sample_set[sample_set[var] > score_median_list[i]]

            left_cnt = left.shape[0]; right_cnt = right.shape[0]
            left1_cnt = sum(left[target]); right1_cnt = sum(right[target])
            left0_cnt = left_cnt - left1_cnt; right0_cnt = right_cnt - right1_cnt
            left_ratio = left_cnt / sample_cnt; right_ratio = right_cnt / sample_cnt

            if left_cnt < min_sample or right_cnt < min_sample:
                continue

            Gini_left = 1 - np.square(left1_cnt / left_cnt) - np.square(left0_cnt / left_cnt)
            Gini_right = 1 - np.square(right1_cnt / right_cnt) - np.square(right0_cnt / right_cnt)
            Gini_temp = Gini - (left_ratio * Gini_left + right_ratio * Gini_right)
            if Gini_temp > bestGini:
                bestGini = Gini_temp; bestSplit_point = score_median_list[i]
                if median_len > 1:
                    bestSplit_position = i / (median_len - 1)
                else:
                    bestSplit_position = i / median_len
            else:
                continue

        Gini = Gini - bestGini
        return bestSplit_point, bestSplit_position

    # 计算最小样本阈值（终止条件）
    min_df = sample_set.shape[0] * 0.05
    split_list = []
    # 计算第一个和最后一个分割点
    bining_data_split(sample_set, var, min_df, split_list,target)
    return split_list


#分组后，根据时间排序后，计算差值；可以计算速度等；
def get_diff(df,key,time='time'):
    import gc
    data = pd.DataFrame()
    for i,ele in enumerate(df.groupby([key])):
        key,value = ele
        value = value.sort_values([time])
        value1 = value.shift(axis=0)
        diff = value-value1
        value = pd.concat([value,diff],axis=1)
        data = pd.concat([data,value])
    #     print(value)
        del value,value1,diff
        gc.collect()
    return data

#cos/sin 将数值的首位衔接起来，比如说 23 点与 0 点很近，星期一和星期天很近,应用于周期
def link_head_tail(df, col, n):
    '''
    data: DataFrame
    col: column name
    n: 时间周期
    '''
    df[col + '_sin'] = round(np.sin(2*np.pi / n * df[col]), 6)
#     df[col + '_cos'] = round(np.cos(2*np.pi / n * df[col]), 6)
    return df
    
class TimeFeatures:
    """
    derived time features,include continuous time and discrete time.
    pandas.Series.dt
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.days_in_month.html
    
    example:
    ------------
    nowtime = 'trans_time'
    starttime = 'create_time'
    tools = Tools()
    results[nowtime] = results[nowtime].apply(lambda x:tools.string2time(x,'%Y-%m-%d %H:%M:%S'))
    results[starttime] = results[starttime].apply(lambda x:tools.string2time(x,'%Y-%m-%d %H:%M:%S'))
    timeFeatures = TimeFeatures()
    results = timeFeatures.tagging_time_fea(results,nowtime,starttime)
    """
    def __init__(self):
        pass
    """
    时间打标
    """
    def tagging_time_fea(self,df,nowtime=None,starttime=None,hour_interval=[8,10],hour_bins=[0,6,12,18,24]):
        if nowtime is not None:
            df[nowtime+'_'+starttime+'_timediff'] = df[nowtime]-df[starttime]
            df[nowtime+'_'+starttime+'_days'] = df[nowtime+'_'+starttime+'_timediff'].dt.days
        df[starttime+'_year'] = df[starttime].dt.year
        df[starttime+'_quarter'] = df[starttime].dt.quarter
        df[starttime+'_month'] = df[starttime].dt.month
        df[starttime+'_day'] = df[starttime].dt.day
        df[starttime+'_dayofweek'] = df[starttime].dt.dayofweek
        df[starttime+'_weekofyear'] = df[starttime].dt.week
        df[starttime+'_weekend'] = df[starttime+'_dayofweek'].apply(lambda x: 1 if x>=5 else 0)
        df[starttime+'_hour'] = df[starttime].dt.hour

        # pandas.Series.dt 下有很多属性，可以去看一下是否有需要的。
        df[starttime+'_is_year_start'] = df[starttime].dt.is_year_start
        df[starttime+'_is_year_end'] = df[starttime].dt.is_year_end
        df[starttime+'_is_quarter_start'] = df[starttime].dt.is_quarter_start
        df[starttime+'_is_quarter_end'] = df[starttime].dt.is_quarter_end
        df[starttime+'_is_month_start'] = df[starttime].dt.is_month_start
        df[starttime+'_is_month_end'] = df[starttime].dt.is_month_end

        # 是否时一天的高峰时段 8~10
        df[starttime+'_is_day_high'] = df[starttime+'_hour'].apply(lambda x: 1 if hour_interval[0] <= x <= hour_interval[1]  else 0)
        # 对小时进行分箱
        df[starttime+'_hour_box'] = pd.cut(df[starttime+'_hour'],bins=hour_bins,right=False)
        df[starttime+'_hour_box'] = df[starttime+'_hour_box'].astype(str).apply(lambda x: re.sub(r'[\[,)]+', "", x).replace(' ','_'))
        
        #上旬
        df['first_ten_days'] = df[starttime+'_day'].apply(lambda x : 1 if x<=10 else 0)
        #中旬
        df['mid_ten_days'] = df[starttime+'_day'].apply(lambda x : 1 if 10<x<=20 else 0)
        #下旬
        df['last_ten_days'] = df[starttime+'_day'].apply(lambda x : 1 if 20<x<=31 else 0)

        df = link_head_tail(df, starttime+'_hour', n=24)
        df = link_head_tail(df, starttime+'_day', n=31)
        df = link_head_tail(df, starttime+'_dayofweek', n=7)
        df = link_head_tail(df, starttime+'_quarter', n=4)
        df = link_head_tail(df, starttime+'_month', n=12)
        df = link_head_tail(df, starttime+'_weekofyear', n=53)

        #节假日、节假日第 n 天、节假日前 n 天、节假日后 n 天
        
        return df

class AggregateCharacteristics:
    """
    example:
    ------------------------
   dayslist = ['all','1','7','30']
   aggregateCharacteristics = AggregateCharacteristics(sample,dayslist
                                          ,continuouscols=['age']
                                          ,discretecols=['marital_status','ktp_gender','city','last_education']
                                          ,combinecols = ['create_time_weekend']
                                          ,continuous_crosscols=[]
                                          ,discrete_crosscols=[['first_ten_days','create_time_weekend'],  
                                                        ['create_time_weekend','create_time_hour_box']]
                                                   )
    aggregateCharacteristics.Aggregate_all_days(df=results,key='customer_id',timediffcol='trans_time_create_time_days')
    aggregateCharacteristics.firsttimeDerived(results,key='customer_id',timecol='create_time',timediffcol='trans_time_create_time_days')
    aggregateCharacteristics.lasttimeDerived(results,key='customer_id',timecol='create_time',timediffcol='trans_time_create_time_days')
    aggregateCharacteristics.trendDerived(cutline=2)
    tmp = aggregateCharacteristics.consistencyCheck(results,'customer_id','gender','ktp_gender')
    """
    
    def __init__(self,sample,dayslist,continuouscols=[],discretecols=[],combinecols=[],continuous_crosscols=[],discrete_crosscols=[],processes=10):
        self.sample = sample.copy()
        self.dayslist = dayslist
        #需要衍生的连续变量
        self.continuouscols = continuouscols
        #需要衍生的离散变量
        self.discretecols = discretecols
        #一维交叉衍生变量
        self.combinecols = combinecols
        #二维交叉衍生变量(笛卡尔乘积)_连续变量聚合
        self.continuous_crosscols = continuous_crosscols
        #二维交叉衍生变量(笛卡尔乘积)_离散变量聚合
        self.discrete_crosscols = discrete_crosscols
        
        #多进程
        self.processes = processes
        
    #最近xx天内
    def Aggregate_all_days(self,df,key,timediffcol=''):
        for ndays in self.dayslist:
            self.Aggregate_days(df,key,timediffcol,ndays)
        
    def Aggregate_days(self,df,key,timediffcol='',ndays='all',smaller=True):
        #ndays is None:所有历史数据参与计算变量
        if ndays=='all':
            self.df_tmp = df
            renamedays = ''
        else:
            self.df_tmp = df[df[timediffcol]<=int(ndays)]
            renamedays = '_last'+ndays+'days'
        
        #连续型变量聚合
        pool = multiprocessing.Pool(processes=self.processes)
        executeResults={}
        for col in self.continuouscols:
            executeResults[col]=pool.apply_async(func=self.continuousAggregate,args=(self.df_tmp,key,col,renamedays))
        pool.close()
        pool.join()
        for k,value in executeResults.items():
            fea=value.get()
            self.sample = pd.merge(self.sample,fea,how='left',on=key)
        #离散型变量聚合
        pool = multiprocessing.Pool(processes=self.processes)
        executeResults={}
        for col in self.discretecols:
            executeResults[col]=pool.apply_async(func=self.discreteAggregate,args=(self.df_tmp,key,col,renamedays))
        pool.close()
        pool.join()
        for k,value in executeResults.items():
            fea=value.get()
            self.sample = pd.merge(self.sample,fea,how='left',on=key)
            
        #一维交叉特征
        pool = multiprocessing.Pool(processes=self.processes)
        executeResults={}
        for discretecol in self.discretecols:
            for combinecol in self.combinecols:
                k = discretecol+combinecol
                executeResults[k]=pool.apply_async(func=self.combine1d_discrete,args=(self.df_tmp,key,combinecol,discretecol,renamedays))
        pool.close()
        pool.join()
        for k,value in executeResults.items():
            results=value.get()
            for fea in results:
                self.sample = pd.merge(self.sample,fea,how='left',on=key)
        
        #笛卡尔乘积交叉特征--离散
        pool = multiprocessing.Pool(processes=self.processes)
        executeResults={}
        for discretecol in self.discretecols:
            for crosscol in self.discrete_crosscols:
                k = discretecol+crosscol[0]+crosscol[1]
                executeResults[k]=pool.apply_async(func=self.cartesianProduct_discrete,args=(self.df_tmp,key,crosscol,discretecol,renamedays))
        pool.close()
        pool.join()
        for k,value in executeResults.items():
            results=value.get()
            for fea in results:
                self.sample = pd.merge(self.sample,fea,how='left',on=key)
        #笛卡尔乘积交叉特征--连续
        for continuouscol in self.continuouscols:
            for crosscol in self.continuous_crosscols:
                pass
    
    #1d衍生-离散
    def combine1d_discrete(self,df,key,combinecol,col,renamedays):
        agg_cols = ['size','count',pd.Series.nunique]
        values = df[combinecol].unique()
        results = []
        for ele in values:
            tmp_df = df[df[combinecol]==ele]
            fea = tmp_df.groupby(key)[col].agg(agg_cols).reset_index()
            fea.rename(columns={'size':combinecol+'_'+str(ele)+'_'+col+'_size'+renamedays,\
                                'count':combinecol+'_'+str(ele)+'_'+col+'_count'+renamedays,\
                                'nunique':combinecol+'_'+str(ele)+'_'+col+'_nunique'+renamedays},inplace=True)
            results.append(fea)
        return results
#             self.sample = pd.merge(self.sample,fea,how='left',on=key)
    
    #1d衍生-连续
    def combine1d_continuous(self,df,key,combinecol,col,renamedays):
        pass
    
    #2d衍生-连续
    def cartesianProduct_continuous(self,df,key,crosscol,col,renamedays):
        pass
    
    #2d衍生-离散
    def cartesianProduct_discrete(self,df,key,crosscol,col,renamedays):
        left_col,right_col = crosscol[0],crosscol[1]
        left_values = df[left_col].unique()
        right_values = df[right_col].unique()
        crossval = [(left, right) for left in left_values for right in right_values]
        agg_cols = ['size','count',pd.Series.nunique]
        results = []
        for left,right in crossval:
            tmp_df = df[(df[left_col]==left)&(df[right_col]==right)]
            fea = tmp_df.groupby(key)[col].agg(agg_cols).reset_index()
            fea.rename(columns={'size':left_col+'_'+str(left)+'_'+right_col+'_'+str(right)+'_'+col+'_size'+renamedays,\
                                'count':left_col+'_'+str(left)+'_'+right_col+'_'+str(right)+'_'+col+'_count'+renamedays,\
                                'nunique':left_col+'_'+str(left)+'_'+right_col+'_'+str(right)+'_'+col+'_nunique'+renamedays},inplace=True)
            results.append(fea)
        return results
#             self.sample = pd.merge(self.sample,fea,how='left',on=key)
            
    def continuousAggregate(self,df,key,col,renamedays,smaller=True):
        if smaller==True:
            agg_cols = ['size','count','min','max','mean','sum','median','std','var']
            rename_dict = {x:col+'_'+x+renamedays for x in agg_cols}
            fea = df.groupby(key)[col].agg(agg_cols).reset_index()
            fea.rename(columns=rename_dict,inplace=True)
            return fea
        else:
            agg_cols = ['size','count','min','max','mean','sum','median','std','mad','var','skew']
            rename_dict = {x:col+'_'+x+renamedays for x in agg_cols}
            fea1 = df.groupby(key)[col].agg(agg_cols).reset_index()
            fea1.rename(columns=rename_dict,inplace=True)
            fea2 = df.groupby(key)[col].apply(pd.Series.kurt).reset_index()
            fea2.rename(columns={col:col+'_kurt'+renamedays},inplace=True)
            fea3 = df.groupby(key)[col].quantile(q=0.25).reset_index()
            fea3.rename(columns={col:col+'_quantile25'+renamedays},inplace=True)
            fea4 = df.groupby(key)[col].quantile(q=0.75).reset_index()
            fea4.rename(columns={col:col+'_quantile75'+renamedays},inplace=True)
            fea = pd.merge(fea1,fea2,how='outer',on=key)
            fea = pd.merge(fea,fea3,how='outer',on=key)
            fea = pd.merge(fea,fea4,how='outer',on=key)
            return fea
        
    def discreteAggregate(self,df,key,col,renamedays):
        agg_cols = ['size','count',pd.Series.nunique]
        fea = df.groupby(key)[col].agg(agg_cols).reset_index()
        fea.rename(columns={'size':col+'_size'+renamedays,'count':col+'_count'+renamedays,'nunique':col+'_nunique'+renamedays},inplace=True)
        return fea
    
    #最近一次
    def lasttimeDerived(self,df,key,timecol='',timediffcol=''):
        df_max = df.groupby(key)[timecol].max().reset_index()
        df_max = pd.merge(df_max,df)
        need_cols = [timediffcol]
        need_cols.extend(self.continuouscols)
        need_cols.extend(self.discretecols)
        rename_dict = {x:'lasttime_'+x for x in need_cols}
        need_cols.append(key)
        fea = df_max[need_cols].rename(columns=rename_dict)
        self.sample = pd.merge(self.sample,fea,how='left',on=key)
    
    #最早一次
    def firsttimeDerived(self,df,key,timecol='',timediffcol=''):
        df_min = df.groupby(key)[timecol].min().reset_index()
        df_min = pd.merge(df_min,df)
        need_cols = [timediffcol]
        need_cols.extend(self.continuouscols)
        need_cols.extend(self.discretecols)
        rename_dict = {x:'firsttime_'+x for x in need_cols}
        need_cols.append(key)
        fea = df_min[need_cols].rename(columns=rename_dict)
        self.sample = pd.merge(self.sample,fea,how='left',on=key)
    
    #最近xx天比近xx天
    def trendDerived(self,cutline=2):
        import re
        pattern = re.compile(r'.+(?=last\d+days)')
        tmp = list(self.sample.columns)
        cols_need = list(set([pattern.search(x).group() for x in tmp if pattern.search(x)]))
        days_list = self.dayslist[1:]
        numerator = days_list[:cutline]
        denominator = days_list[cutline:]
        trenddays = [(x,y) for x in numerator for y in denominator]
        for col in cols_need:
            for days1,days2 in trenddays:
                numerator_col = col +'last'+days1+'days'
                denominator_col = col +'last'+days2+'days'
                self.sample[col+'last'+days1+'_div_'+days2+'days'] = self.sample.apply(lambda x:100.0*x[numerator_col]/x[denominator_col] if x[denominator_col]!=0 else -9999976,axis=1)
                self.sample[col+'last'+days2+'_sub_'+days1+'days'] = self.sample.apply(lambda x:x[denominator_col]-x[numerator_col],axis=1)
        
    
    #个体与整体差异
    def getIndividual_global_differences(self):
        pass
    
    #数据尺度转换
    def transform_log(self):
        pass
    
    def consistencyCheck(self,df,key,leftcol,rightcol):
        renamecol = leftcol+'_'+rightcol+'_isConsistency'
        df[renamecol] = df.apply(lambda x: 1 if str(x[leftcol])==str(x[rightcol]) else 0,axis=1)
        fea = df.groupby(key)[renamecol].min().reset_index()
        return fea
     
    

from category_encoders import *
class CategoryEncoders:
    '''
    category_encoders applying
    '''
    def __init__(self,encoder=1):
        encoders_dict = { 1:BinaryEncoder,
                    2:OneHotEncoder
                    }
        
        self.encoder = encoders_dict.get(encoder)
    
    def applying_categoryEncoders(self,train_df,test_df,cols):
        '''
        处理类别变量 category_encoders库里封装了大量的方法，详见： https://github.com/scikit-learn-contrib/category_encoders
        '''
        #enc = BinaryEncoder(cols=cat_cols).fit(X_train)
        #直接cat_cols=cat_cols超出内存，用for代替
        for col in cols:
            enc = self.encoder(cols=col).fit(train_df)
            train_df = enc.transform(train_df)
            test_df = enc.transform(test_df)
        return train_df,test_df
    

#%% woe分箱, iv and transform
def get_iv(df,cols,target,outputfile='./data/feature_detail_iv_list.csv'):
    import woe.feature_process as fp
    import woe.eval as eval
    # 分别用于计算连续变量与离散变量的woe。它们的输入形式相同：
    # proc_woe_discrete(df,var,global_bt,global_gt,min_sample,alpha=0.01)
    # # proc_woe_continuous(df,var,global_bt,global_gt,min_sample,alpha=0.01)
    # 输入：
    # df: DataFrame，要计算woe的数据，必须包含'target'变量，且变量取值为{0，1}
    # var:要计算woe的变量名
    # global_bt:全局变量bad total。df的正样本数量
    # global_gt:全局变量good total。df的负样本数量
    # min_sample:指定每个bin中最小样本量，一般设为样本总量的5%。
    # alpha:用于自动计算分箱时的一个标准，默认0.01.如果iv_划分>iv_不划分*（1+alpha)则划分。
    data = df.copy()
    data_woe = data
    data_woe.rename(columns={target:'target'},inplace=True)
    #用于存储所有数据的woe值
    civ_list = []
    n_positive = sum(data['target'])
    n_negtive = len(data) - n_positive
    for column in list(cols):
        if data[column].dtypes == 'object' or 'category':
            civ = fp.proc_woe_discrete(data, column, n_positive, n_negtive, 0.05*len(data), alpha=0.05)
        else:            
            civ = fp.proc_woe_continuous(data, column, n_positive, n_negtive, 0.05*len(data), alpha=0.05)
        civ_list.append(civ)
        data_woe[column] = fp.woe_trans(data[column], civ)

    civ_df = eval.eval_feature_detail(civ_list,outputfile)
    #删除iv值过小的变量
#     iv_thre = 0.001
#     iv = civ_df[['var_name','iv']].drop_duplicates()
#     x_columns = iv.var_name[iv.iv > iv_thre]
    return civ_df

class CustomFeatureSelector:
    '''
    自定义特征选择
     '''
    def __init__(self):
        pass
    
    def randomForestSelectFeas(self,X,y,is_fit=True):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.externals import joblib
        if is_fit:
            randomforest_model = RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=2019).fit(X,y)
            joblib.dump(randomforest_model,'./data/randomforest_model.pkl')
            randomforest_feat_importances = pd.Series(randomforest_model.feature_importances_, index= X.columns)
            randomforest_feat_importances = pd.DataFrame(randomforest_feat_importances).reset_index()
            randomforest_feat_importances.columns = ['feature','important']
            randomforest_feat_importances = randomforest_feat_importances.sort_values(['important'],ascending=False)
            return randomforest_feat_importances
        else:
            randomforest_feas = joblib.load('./data/randomforest_model.pkl')
            randomforest_feat_importances = pd.Series(randomforest_feas.feature_importances_, index= X.columns)
            randomforest_feat_importances = pd.DataFrame(randomforest_feat_importances).reset_index()
            randomforest_feat_importances.columns = ['feature','important']
            randomforest_feat_importances = randomforest_feat_importances.sort_values(['important'],ascending=False)
            return randomforest_feat_importances