from .publicLibrary import *

print('you have imported dataPreprocess')

def get_csv_file_df(train_path,test_path=None,sep=','):   
    train = pd.read_csv(train_path,sep=sep)
    if test_path is not None:
        test = pd.read_csv(test_path,sep=sep)
        return train,test
    else:
        return train,pd.DataFrame()
    

def json2dataframe_resolver(x,columns=[],col=''):
    import time
    import gc
    import multiprocessing
    def json_resolver(x):
        results = []
        tmp = json.loads(x[col])
        for ele in tmp:
            context = ele.get(col,np.nan)
            result = [x[ele] for ele in columns[:-1]]
            result.append(context)
            results.append(result)
        return pd.DataFrame(results,columns=columns)

    start = time.time()
    pool = multiprocessing.Pool(processes=10)
    executeResults={}
    wifi_list = wifi_list.reset_index(drop=True)
    for idx in range(wifi_list.shape[0]):
        key = str(idx)
        executeResults[key] = pool.apply_async(func=json_resolver,args=(wifi_list.iloc[idx],columns,col))
    pool.close()
    pool.join()
    results = pd.DataFrame()
    for k,value in executeResults.items():
        result = value.get()
        results = pd.concat([results,result])
    del executeResults
    gc.collect()

    end = time.time()
    print("Execution Time: ", end - start)