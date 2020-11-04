print("you have imported modelMethod")
from .publicLibrary import *

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score,classification_report,roc_curve,auc,accuracy_score

class BayesoptModel:
    """
    example
    --------------
    model_params = {
            'learning_rate':(0.05, 0.1),
            'max_bin':(10,255),
            'num_leaves':(10,35),
            'n_estimators': (100, 300),
            'max_depth': (2, 13),
            'min_split_gain':(1,5),
            'colsample_bytree':(0.9,1.0),
            'subsample':(0.5,1.0),
            'reg_alpha':(0.1,5.0),
            'reg_lambda':(100,800),
            'min_child_weight':(0.01,0.1),
            'min_child_samples':(10,100)
        }
    bayesoptModel = modelMethod.BayesoptModel(x,y,model_type='LGBMClassifier',scoring='roc_auc',cv=5)
    model_bo_max = bayesoptModel.start_bayes_opt(model_params)
    """
    def __init__(self,x,y,model_type='LGBMClassifier',scoring='roc_auc',cv=5):
        self.x = x
        self.y = y
        self.scoring = scoring
        self.cv = cv
        
        #dict替代if-else
        modeltype_dict = {'LogisticRegression':self.model_cv_LogisticRegression,
                    'SVC':self.model_cv_SVC,
                    'LGBMClassifier':self.model_cv_LGBMClassifier,
                    'RandomForestClassifier':self.model_cv_RandomForestClassifier,
                    'XGBClassifier':self.model_cv_XGBClassifier,
                    'CatBoostClassifier':self.model_cv_CatBoostClassifier}
        
        self.model_cv = modeltype_dict.get(model_type)
        
    def model_cv_LogisticRegression(self,tol,C,max_iter):
        val = cross_val_score(
            LogisticRegression(
                tol=tol,
                C=C,
                max_iter=int(max_iter),
                penalty='l2',
                dual=False,
                fit_intercept=True,
                intercept_scaling=1,
                class_weight='balanced',
                random_state=2020,
                solver='lbfgs',
                multi_class='auto',
                verbose=0,
                warm_start=False,
                n_jobs=None,
                l1_ratio=None
            ),self.x, self.y, scoring=self.scoring, cv=self.cv
        ).mean()
        return val
    
    def model_cv_SVC(self,C):
        val = cross_val_score(
            SVC(
                C=C,
                kernel='rbf',
                degree=3,
                gamma='scale',
                coef0=0.0,
                shrinking=True,
                probability=False,
                tol=0.001,
                cache_size=1000,
                class_weight='balanced',
                verbose=False,
                max_iter=-1,
                decision_function_shape='ovr',
                break_ties=False,
                random_state=2020,
            ),self.x, self.y, scoring=self.scoring, cv=self.cv
        ).mean()
        return val
    
    
    def model_cv_LGBMClassifier(self,learning_rate, max_bin, num_leaves, n_estimators, max_depth
                 ,min_split_gain,colsample_bytree
                 ,subsample,reg_alpha,reg_lambda,min_child_weight,min_child_samples
                ):
        val = cross_val_score(
            LGBMClassifier(
                learning_rate = learning_rate,
                max_bin = int(max_bin),
                num_leaves = int(num_leaves),
                n_estimators = int(n_estimators),
                max_depth=int(max_depth),
                min_split_gain=int(min_split_gain),
                colsample_bytree=colsample_bytree,
                subsample=subsample,
                reg_alpha=reg_alpha,
                reg_lambda=int(reg_lambda),
                min_child_weight=min_child_weight,
                min_child_samples=int(min_child_samples),

                random_state=2020,
                is_unbalance=True,
                objective='binary'
    #             objective = 'multiclass',
    #             num_class = 2,
            ),self.x, self.y, scoring=self.scoring, cv=self.cv
        ).mean()
        return val
    
    def model_cv_RandomForestClassifier(self,n_estimators,max_depth,min_samples_split,min_samples_leaf):
        val = cross_val_score(
            RandomForestClassifier(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                min_samples_split = int(min_samples_split),
                min_samples_leaf= int(min_samples_leaf),
                min_weight_fraction_leaf=0.0,
                criterion='gini',
                max_features='auto',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                min_impurity_split=None,
                bootstrap=True,
                oob_score=True,
                n_jobs=None,
                random_state=2020,
                verbose=0,
                warm_start=False,
                class_weight='balanced',
                ccp_alpha=0.0,
                max_samples=None,
            ),self.x, self.y, scoring=self.scoring, cv=self.cv
        ).mean()
        return val
    
    def model_cv_XGBClassifier(self,learning_rate,n_estimators,max_depth,gamma,
                          subsample,colsample_bytree,reg_alpha,reg_lambda):
        val = cross_val_score(
            XGBClassifier(
                learning_rate = learning_rate,
                n_estimators = int(n_estimators),
                max_depth=int(max_depth),
                gamma = gamma,
#                 min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha= reg_alpha,
                reg_lambda = reg_lambda,
                
                random_state=2020,
#                 scale_pos_weight=1,
#                 objective='binary'
    #             objective = 'multiclass',
    #             num_class = 2,
            ),self.x, self.y, scoring=self.scoring, cv=self.cv
        ).mean()
        return val
    
    def model_cv_CatBoostClassifier(self,depth,n_estimators,l2_leaf_reg,subsample):
        val = cross_val_score(
            CatBoostClassifier(
                depth=int(depth),
                n_estimators=int(n_estimators),
                l2_leaf_reg=l2_leaf_reg,
                subsample=subsample,
                
                one_hot_max_size=2,
#                 class_weights=None,
                random_state=2020,
#                 scale_pos_weight=1,
#                 objective='binary'
    #             objective = 'multiclass',
    #             num_class = 2,
            ),self.x, self.y, scoring=self.scoring, cv=self.cv
        ).mean()
        return val
    
    
    
    def start_bayes_opt(self,model_params):
        model_bo = BayesianOptimization(
            self.model_cv,
            model_params
            )
        model_bo.maximize()
        return model_bo.max

    
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.plotting import plot_decision_regions
class EnsembleMethods:
    """
    example
    --------------
    
    """
    def __init__(self,models=[],):
        pass
    