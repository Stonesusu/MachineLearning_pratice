print("you have imported customSample")
from .publicLibrary import *

def random_negative_sampling(train,test=None,sample_size=10000):
    sample_train = train.sample(sample_size)
    if test is not None:
        sample_test = test.sample(sample_size)
        return sample_train,sample_test
    else:
        return sample_train,pd.DataFrame()
    

def custom_negative_sampling(df,target,sample_dict):
    result = []
    for k,v in sample_dict.items():
        tmp = df[df[target]==k].sample(frac=v)
        result.append(tmp)
    result = pd.concat(result)
    return result

#待增加smote等方法