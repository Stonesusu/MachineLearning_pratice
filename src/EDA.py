print("you have imported EDA")
from .publicLibrary import *

#分析label占比
def label_analysis(df,target):
    rate = pd.DataFrame(df[target].value_counts())
    rate['rate'] = 100.0*rate[target].apply(lambda x: x/df.shape[0])
    return rate

def get_index_null(df,threshold=0.8,axis=1):
    if axis==1:
        col_null = df.isnull().sum(axis=0)
        col_null = col_null[col_null>threshold*df.shape[0]]
        return list(col_null.index)
    else:
        row_null = df.isnull().sum(axis=1)
        row_null = row_null[row_null>threshold*df.shape[1]]
        return list(row_null.index)
    
def null_fea_analysis(df,target,key,threshold=0.8):
    results = []
    col_nan = get_index_null(df,threshold=threshold,axis=1)
    for col in col_nan:
        tmp = df.copy()
        tmp = tmp.replace(np.nan,-99999)
        tmp = pd.pivot_table(tmp,index=col,columns=target,values=key,aggfunc=len,fill_value=0,margins=True,margins_name='All')
        tmp['rate'] = 100.0*tmp['All']/tmp.loc['All']['All']
        tmp['1_rate'] = 100.0*tmp[1]/tmp['All']
        results.append(tmp)
        print(tmp)
        print('**************************************************')
    return results

def fea_kde_plot(train,test,fea_cols,target,target_list,figsize=(15,5)):
    import gc
    #kde,如果变量很多，那么先模型筛再kde，如果变量不多，那么先kde，再模型筛
    def plot_kde(train, test, col,target='label',target_list=None,values=True):
        fig,ax =plt.subplots(1,4,figsize=figsize)
        if target_list is not None:
            colors = ['g','r','y','b']
        for i,ele in enumerate(target_list):
            sns.kdeplot(train[col][train[target]==ele],color=colors[i],ax=ax[0],label='label_'+str(ele))

        sns.kdeplot(train[col],color='y',ax=ax[1],label='train')

        sns.kdeplot(test[col],color='b',ax=ax[2],label='test')

        sns.kdeplot(train[col],color='y',ax=ax[3],label='train')
        sns.kdeplot(test[col],color='b',ax=ax[3],label='test')
        plt.xlabel(col,size=16)
        plt.show()
        del train,col,test
        gc.collect()
    for col in fea_cols:
        plot_kde(train, test, col,target=target,target_list=target_list,values=True)
        



    
#柱状图
def bar_plot(df,x,y,hue=None,figsize=(16,5),rotation=0,order=None,orient=None):
    f, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=x,y=y,data=df,hue=hue,order=order,orient=orient)
    plt.xticks(rotation=rotation)
    

#单维直方图，附加kde
def dist_plot(df,col,figsize=(8, 7)):
    sns.set_style("white")
    sns.set_color_codes(palette='deep')
    f, ax = plt.subplots(figsize=figsize)
    #Check the new distribution 
    sns.distplot(df[col], color="b");
    ax.xaxis.grid(False)
    ax.set(ylabel="Frequency")
    ax.set(xlabel=col)
    ax.set(title=col+" distribution")
    sns.despine(trim=True, left=True)
    plt.show()
    
#盒图
def box_plot_1d(df,x,figsize=(16,5),orient='h'):
    sns.set_style("white")
    f, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")  ##数据量纲太大，log解决
    plt.xlabel(x)
    plt.ylabel("value")
    sns.boxplot(data=df[x] , orient=orient, palette="Set1")
    ax.xaxis.grid(False)
    ax.set(ylabel="Feature names")
    ax.set(xlabel="values")
    ax.set(title="Distribution of Features")
    sns.despine(trim=True, left=True)
    
def box_plot_multi_dim(df,x,y=None,hue=None,figsize=(16,5),rotation=45,orient=None):
    f, ax = plt.subplots(figsize=figsize)
    fig = sns.boxplot(x=x, y=y,hue=hue,orient=orient,data=df)
#     fig.axis(ymin=0, ymax=800000);
    plt.xticks(rotation=rotation)
    
def scatter_plot(df,x,y,alpha=0.3, ylim=(0,800000)):
    df.plot.scatter(x=x, y=y, alpha=alpha, ylim=ylim);

def heatmap_plot(df,vmax=1.0,figsize=(15,12),cmap='Blues',square=True):
    corr = df.corr()
    plt.subplots(figsize=figsize)
    sns.heatmap(corr, vmax=vmax, cmap=cmap, square=square)