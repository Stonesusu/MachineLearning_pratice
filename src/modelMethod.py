print("you have imported modelMethod")
from .publicLibrary import *


from lightgbm import LGBMClassifier
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
    bayesoptModel = modelMethod.BayesoptModel(x,y,scoring='roc_auc',cv=5)
    model_bo_max = bayesoptModel.start_bayes_opt(model_params)
    """
    def __init__(self,x,y,scoring='roc_auc',cv=5):
        self.x = x
        self.y = y
        self.scoring = scoring
        self.cv = cv
    
    def model_cv(self,learning_rate, max_bin, num_leaves, n_estimators, max_depth
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

                random_state=2,
                is_unbalance=True,
                objective='binary'
    #             objective = 'multiclass',
    #             num_class = 2,
            ),
            self.x, self.y, scoring=self.scoring, cv=self.cv
        ).mean()
        return val
    
    def start_bayes_opt(self,model_params):
        model_bo = BayesianOptimization(
            self.model_cv,
            model_params
            )
        model_bo.maximize()
        return model_bo.max