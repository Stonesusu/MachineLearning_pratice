from .publicLibrary import *

print('you have imported fileOperation')

def get_csv_file_df(train_path,test_path=None,sep=','):   
    train = pd.read_csv(train_path,sep=sep)
    if test_path is not None:
        test = pd.read_csv(test_path,sep=sep)
        return train,test
    else:
        return train,pd.DataFrame()