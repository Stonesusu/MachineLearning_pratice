print("you have imported modelExplainability")
from .publicLibrary import *
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
def genderate_PermutationImportance(X_train,y_train,is_test=True):
    import eli5
    from eli5.sklearn import PermutationImportance
    if is_test==False:
#         model = LGBMClassifier(**self.params).fit(X_train,y_train)
        model = RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=2019).fit(X_train,y_train)
        perm_train = PermutationImportance(model,random_state=1).fit(X_train,y_train)
    #         eli5.show_weights(perm_train,top=100,feature_names=X_train.columns.tolist())
    #         eli5.show_weights(perm_test,top=100,feature_names=X_test.columns.tolist())
        perm_feature_importance_train = pd.concat([pd.Series(X_train.columns),pd.Series(perm_train.feature_importances_)],axis=1).sort_values(by=1,ascending=False)
        perm_feature_importance_train.columns = ['feature','imp']
        perm_feature_importance_train = perm_feature_importance_train.reset_index(drop=True)
        perm_feature_importance_train.to_csv('../data/perm_feature_importance_train.csv',index=False)

    if is_test==True:
        X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.3)
        model = RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=2019).fit(X_train,y_train)
        perm_train = PermutationImportance(model,random_state=1).fit(X_train,y_train)
    #         eli5.show_weights(perm_train,top=100,feature_names=X_train.columns.tolist())
    #         eli5.show_weights(perm_test,top=100,feature_names=X_test.columns.tolist())
        perm_feature_importance_train = pd.concat([pd.Series(X_train.columns),pd.Series(perm_train.feature_importances_)],axis=1).sort_values(by=1,ascending=False)
        perm_feature_importance_train.columns = ['feature','imp']
        perm_feature_importance_train = perm_feature_importance_train.reset_index(drop=True)
        perm_feature_importance_train.to_csv('./data/perm_feature_importance_train.csv',index=False)
        perm_test = PermutationImportance(model,random_state=1).fit(X_test,y_test)
        perm_feature_importance_test = pd.concat([pd.Series(X_test.columns),pd.Series(perm_test.feature_importances_)],axis=1).sort_values(by=1,ascending=False)
        perm_feature_importance_test.columns = ['feature','imp']
        perm_feature_importance_test = perm_feature_importance_test.reset_index(drop=True)
        perm_feature_importance_test.to_csv('./data/perm_feature_importance_test.csv',index=False)
        
def get_permutation_feature_imp(is_test=False,imp_threshold=0,how_param = 'outer'):
    perm_feature_importance_train = pd.read_csv('./data/perm_feature_importance_train.csv')
    perm_feature_importance_train = perm_feature_importance_train[perm_feature_importance_train['imp']>imp_threshold]

    # 挑选出train、valid中变量重要性>0的变量
    if is_test:
        perm_feature_importance_test = pd.read_csv('./data/perm_feature_importance_test.csv')
        perm_feature_importance_test = perm_feature_importance_test[perm_feature_importance_test['imp']>imp_threshold]
        perm_feature_select = pd.merge(perm_feature_importance_train,perm_feature_importance_test,how=how_param,on='feature')
        #合并变量集，可以尝试不同的合并方式，放入模型对比效果；
    else:
        perm_feature_select = perm_feature_importance_train
    return perm_feature_select

import shap
import multiprocessing
class ShapTreeExplainer:
    def __init__(self,model,df,feature_perturbation='interventional',sample_frac=None):
        self.df = df
        if sample_frac is not None:
            self.df = self.df.sample(frac=sample_frac,random_state=2020)
        self.df = self.df.reset_index(drop=True)
        self.model = model
        self.explainer = shap.TreeExplainer(self.model,self.df,feature_perturbation=feature_perturbation)
        shap.initjs()  # notebook环境下，加载用于可视化的JS代码
        
    def shap_force_plot(self,index,approximate=False):
        data_test = self.df.iloc[index].astype('float')
        shap_values = self.explainer.shap_values(data_test,approximate=approximate)
        shap.force_plot(self.explainer.expected_value[1], shap_values[1], data_test)
    
    def get_data_shapvalues(self,approximate=False,processes=None):
        if processes is None:
            self.shap_values = self.explainer.shap_values(self.df,approximate=approximate)
        else:
            # calculate shap values with multi-process mode
            pool = multiprocessing.Pool(processes=processes)
            executeResults={}
            for idx in range(self.df.shape[0]):
                data_test = self.df.iloc[idx].astype('float')
                executeResults[idx]=pool.apply_async(func=self.explainer.shap_values,args=(data_test,None,None,approximate))
            pool.close()
            pool.join()
            shap_values = []
            shap_value0 = []
            shap_value1 = []
            list1= sorted(executeResults.items(),key=lambda x:x[0])
            for k,value in list1:
                fea=value.get()
                shap_value0.append(fea[0])
                shap_value1.append(fea[1])
            shap_value0 = np.array(shap_value0)
            shap_value1 = np.array(shap_value1)
            shap_values.append(shap_value0)
            shap_values.append(shap_value1)
            self.shap_values = shap_values
    
    def shap_summary_plot(self,rows,max_display=50):
        # calculate shap values. This is what we will plot.
        # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
        # shap_values = self.explainer.shap_values(self.df[:rows],approximate=approximate)
        # Make plot. Index of [1] is explained in text below.
        shap.summary_plot(self.shap_values[1][:rows], self.df[:rows],max_display=max_display)
    
    def shap_dependence_plot(self,ind='',interaction_index='auto'):
        # dependence_plot
        # 为了理解单个feature如何影响模型的输出，我们可以将该feature的SHAP值与数据集中所有样本的feature值进行比较。
        # 由于SHAP值表示一个feature对模型输出中的变动量的贡献，下面的图表示随着特征RM变化的预测房价(output)的变化。
        # 单一RM(特征)值垂直方向上的色散表示与其他特征的相互作用，为了帮助揭示这些交互作用，“dependence_plot函数”
        # 自动选择另一个用于着色的feature。在这个案例中，RAD特征着色强调了RM(每栋房屋的平均房间数)对RAD值较高地区的房价影响较小。
        # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
        # interaction_index :“auto”, None, int, or string
        shap.dependence_plot(ind=ind, shap_values=self.shap_values[1], features=self.df,interaction_index=interaction_index)

    """还有更多功能：
    鼠标可以放图上面显示具体数值
    shap.force_plot(explainer.expected_value[1], shap_values[1], val_X[:10000])
    
    #Feature Importance
    # SHAP提供了另一种计算特征重要性的思路。
    # 取每个特征的SHAP值的绝对值的平均值作为该特征的重要性，得到一个标准的条形图(multi-class则生成堆叠的条形图)
    shap.summary_plot(shap_values[1], val_X[:10000], plot_type="bar",layered_violin_max_num_bins=20)
    
    # interaction value是将SHAP值推广到更高阶交互的一种方法。
    # 实现了快速、精确的两两交互计算，这将为每个预测返回一个矩阵，其中主要影响在对角线上，交互影响在对角线外。
    # 这些数值往往揭示了有趣的隐藏关系(交互作用)
    shap_interaction_values = explainer.shap_interaction_values(X)
    shap.summary_plot(shap_interaction_values, X)
    
    
    """