import pandas as pd
import numpy as np
import lightgbm as lgb
import time,datetime
from sklearn.model_selection import KFold, cross_val_score, train_test_split,RepeatedKFold
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import catboost as cb
import xlearn as xl
from sklearn.linear_model import LinearRegression,BayesianRidge
import gc
import warnings
warnings.filterwarnings('ignore')

#Reduce the memory usage - Inspired by Panchajanya Banerjee
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def get_cross_feature(data):
    cross_features = []
    A_features = ['A6','A7','A15','A19','A20','A25','A27']
    B_features = ['B1','B6','B7','B9','B10','B11','B12','B14']
    for i in range(len(A_features)-1):
        for j in range(i+1,len(A_features)):
            col_name = "cross_" + A_features[i] + "_and_" + A_features[j]
            cross_features.append(col_name)
            data[col_name] = data[A_features[i]].astype(str).values + '_' + data[A_features[j]].astype(str).values

    for i in range(len(B_features)-1):
        for j in range(i+1,len(B_features)):
            col_name = "cross_" + B_features[i] + "_and_" + B_features[j]
            cross_features.append(col_name)
            data[col_name] = data[B_features[i]].astype(str).values + '_' + data[B_features[j]].astype(str).values

    for f in cross_features:
        data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
    
    data = data[cross_features]
    return data



def timeTranSecond(t):
    tm = 0
    try:
        t,m,s=t.split(":")
    except:
        if t=='1900/1/9 7:00':
            return 7*3600/3600
        elif t=='1900/1/1 2:30':
            return (2*3600+30*60)/3600
        elif t==-1:
            return -1
        else:
            return 0
    
    try:
        tm = (int(t)*3600+int(m)*60+int(s))/3600
    except:
        return (30*60)/3600
    
    return tm


def timestart1(t):
    tm = 0
    try:
        t,m,s=t.split(":")
    except:
        if t=='1900/1/9 7:00':
            tm = 7
        elif t=='1900/1/1 2:30':
            tm = 2
        elif t==-1:
            tm = -1
        else:
            tm = 0
    
    try:
        tm = int(t) + int(m)/60
    except:
        tm = -1

    if(tm >=0 and tm<6):
        return 1
    elif(tm >=6 and tm<12):
        return 2
    elif(tm >=12 and tm<18):
        return 3
    else:
        return 4

def timestart2(se):

    tm = 0 
    try:
        sh,sm,eh,em=re.findall(r"\d+\.?\d*",se)
    except:
        if se == -1:
            tm = -1 
        
    try:
        tm = int(sh)+int(sm)/60
    except:
        if se=='19:-20:05':
            tm = 19
        elif se=='15:00-1600':
            tm = 15
        else:
            tm = -1

    return tm
    '''
    if(tm >=0 and tm<6):
        return 1
    elif(tm >=6 and tm<12):
        return 2
    elif(tm >=12 and tm<18):
        return 3
    else:
        return 4
    '''

def getDuration(se):

    tm = 0 
    try:
        sh,sm,eh,em=re.findall(r"\d+\.?\d*",se)
    except:
        if se == -1:
            return -1 
        
    try:
        if int(sh)>int(eh):
            tm = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600 + 24
        else:
            tm = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600
    except:
        if se=='19:-20:05':
            return 1
        elif se=='15:00-1600':
            return 1
        else:
            return -1
    
    return tm
    

def time_diff(time1,time2):
    try:
        t1,m1,s1 = time1.split(":")
        t2,m2,s2 = time2.split(':')
    except:
        if (time1=='1900/1/9 7:00') or (time2=='1900/1/9 7:00') :
            return 90
        elif (time1=='1900/1/1 2:30') or (time2=='1900/1/1 2:30'):
            return 90
        elif (time1==-1) or(time2 == -1):
            return -1
        else:
            return 90
    
    try:
        if(t1 > t2):
            time_differ = ((int(t2) + 24)*60 + int(m2))- (int(t1)*60 + int(m1))
        else:
            time_differ = (int(t2)*60 + int(m2))- (int(t1)*60 + int(m1))
    except:
        return 90
    
    return time_differ

###第四部分 cross feature:
def get_cross_feature(data):
    cross_features = []
    A_features = ['A6','A7','A15','A19','A20','A25','A27']
    B_features = ['B1','B6','B7','B9','B10','B11','B12','B14']
    for i in range(len(A_features)-1):
        for j in range(i+1,len(A_features)):
            col_name = "cross_" + A_features[i] + "_and_" + A_features[j]
            cross_features.append(col_name)
            data[col_name] = data[A_features[i]].astype(str).values + '_' + data[A_features[j]].astype(str).values

    for i in range(len(B_features)-1):
        for j in range(i+1,len(B_features)):
            col_name = "cross_" + B_features[i] + "_and_" + B_features[j]
            cross_features.append(col_name)
            data[col_name] = data[B_features[i]].astype(str).values + '_' + data[B_features[j]].astype(str).values
    return data,cross_features

def encode_count(df,column_name):
    lbl = LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name] = lbl.transform(list(df[column_name].values))
    return df

def encode_onehot(df,column_name):
    data_spr = sparse.csr_matrix((len(df), 0))
    enc = OneHotEncoder()
    for f in column_name:
        df[f] = df[f].map(dict(zip(df[f].unique(), range(0, df[f].nunique()))))
    for f in column_name:
        enc.fit(df[f].values.reshape(-1, 1))
        data_spr = sparse.hstack((data_spr, enc.transform(df[f].values.reshape(-1, 1))), 'csr','float32')
    return data_spr

def baseline_para_lgb(X_train,y_train,X_test,test,cate_list = [],seed = 2018,round = 5000,n_folds = 5,output = True):
    res = test.copy()
    features = X_train.columns
    feature_importance_df = pd.DataFrame()
    param = {
             'num_leaves': 120,
             'min_data_in_leaf': 10, 
             'objective':'regression',
             'max_depth': -1,
             'learning_rate': 0.01,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9,
             "bagging_seed": 11,
             "metric": 'mse',
             "lambda_l1": 0.1,
             "verbosity": -1,
            }
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=2018)
    oof_lgb = np.zeros(X_train.shape[0])
    predictions_lgb = np.zeros(X_test.shape[0])

    for index, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("Fold:", index+1)
        trn_data = lgb.Dataset(X_train.iloc[trn_idx], y_train[trn_idx])##,categorical_feature = [i for i in X_train.columns if 'cross' in i]
        val_data = lgb.Dataset(X_train.iloc[val_idx], y_train[val_idx])

        clf = lgb.train(param, trn_data, round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)#,categorical_feature = cate_list
        oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)

        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = index + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


    cross_validation_loss = mean_squared_error(y_train,oof_lgb)
    print('Training set and test set shape：',X_train.shape,X_test.shape) 
    print('MSE:', cross_validation_loss)
    if(output):
        res[1] = predictions_lgb
        mean = res[1].mean()
        print('mean:',mean)
        res.to_csv("./lgb_base.csv", index=False,header = None)
      
        return res,oof_lgb,feature_importance_df,cross_validation_loss
    else:
        return predictions_lgb,oof_lgb,cross_validation_loss
 
def get_diff(data):
    new_data = data.copy()
    copy_col = ['B12', 'B14','A6', 'B9', 'B10', 'B11']
    keep_col = ['样本id']
    new_data = new_data.sort_values(keep_col, ascending=True)
    new_data['b14_div_a1_a2_a3_a4_a19_b1b2_b12b13'] = new_data['B14']/(new_data['A1']+new_data['A2']+new_data['A3']+new_data['A4']+new_data['A19']+new_data['B1']*new_data['B2']+new_data['B12'])

    # all.loc[all['B14']<=360, 'B14'] = all['B14']+140
    ids = new_data['样本id'].values

    all_copy_previous_row = new_data[copy_col].copy()
    all_copy_previous_row['time_mean'] = all_copy_previous_row[['B9','B10','B11']].std(axis=1)
    all_copy_previous_row.drop(['B9','B10','B11'], axis=1, inplace=True)
    all_copy_previous_row = all_copy_previous_row.diff(periods=1)
    all_copy_previous_row.columns = [col_+'_difference'+'_previous' for col_ in all_copy_previous_row.columns.values]
    all_copy_previous_row['样本id'] = list(ids)    
    print(all_copy_previous_row.columns)
    all_copy_following_row = new_data[copy_col].copy()
    all_copy_following_row['time_mean'] = all_copy_following_row[['B9','B10','B11']].std(axis=1)
    all_copy_following_row.drop(['B9','B10','B11'], axis=1, inplace=True)
    all_copy_following_row = all_copy_following_row.diff(periods=-1)
    all_copy_following_row.columns = [col_+'_difference'+'_following' for col_ in all_copy_following_row.columns.values]
    # all_copy_following_row['样本id_difference_following'] = all_copy_following_row['样本id_difference_following'].abs()
    all_copy_following_row['样本id'] = list(ids)
    new_data = pd.merge(new_data[['样本id']],all_copy_previous_row,on = ['样本id'],how = 'left')
    new_data = pd.merge(new_data,all_copy_following_row,on = ['样本id'],how = 'left')
    return new_data


def get_cross_feature_new(new_data):
    data = new_data.copy()
    cross_features = []
    A_features = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A14',
                  'A15','A16','A17','A19','A20','A21','A22','A24','A25','A26','A27','A28']
    B_features = ['B1','B2','B4','B5','B6','B7','B9','B10','B11','B12','B14']
    data = data[A_features + B_features]
    for i in range(len(A_features)-1):
        for j in range(i+1,len(A_features)):
            col_name = "cross_" + A_features[i] + "_and_" + A_features[j]
            cross_features.append(col_name)
            data[col_name] = data[A_features[i]].astype(str).values + '_' + data[A_features[j]].astype(str).values
            '''
            df = data.groupby(A_features[i])[col_name].count().reset_index()
            df.columns = [A_features[i],A_features[i] + '_%s_count'%col_name]
            data = data.merge(df,on = [A_features[i]],how = 'left')
            df = data.groupby(A_features[j])[col_name].count().reset_index()
            df.columns = [A_features[j],A_features[i] + '_%s_count'%col_name]
            data = data.merge(df,on = [A_features[j]],how = 'left')
            '''

    for i in range(len(B_features)-1):
        for j in range(i+1,len(B_features)):
            col_name = "cross_" + B_features[i] + "_and_" + B_features[j]
            cross_features.append(col_name)
            data[col_name] = data[B_features[i]].astype(str).values + '_' + data[B_features[j]].astype(str).values
            
            
    for i in A_features:
        for j in B_features:
            col_name = "cross_" + i + "_and_" + j
            cross_features.append(col_name)
            data[col_name] = data[i].astype(str).values + '_' + data[j].astype(str).values
            
    
    for f in cross_features:
        data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
    gc.collect()
    
    data = data.drop(B_features + A_features,axis = 1)
    return data



def baseline_para_xgb(X_train,y_train,X_test,test,seed = 2018,round = 5000,n_folds = 5,output = True):
    
    res = test.copy()
    xgb_params = {'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}

    folds = KFold(n_splits=n_folds, shuffle=True, random_state=2018)
    oof_xgb = np.zeros(X_train.shape[0])
    predictions_xgb = np.zeros(X_test.shape[0])
    for index, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("Fold:", index+1)
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

        clf = xgb.train(dtrain=trn_data, num_boost_round=round, evals = [(trn_data, 'train'), (val_data, 'valid_data')], early_stopping_rounds=200, verbose_eval=100, params=xgb_params)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)

        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
    print('Training set and test set shape：',X_train.shape,X_test.shape) 

    cross_validation_loss = mean_squared_error(y_train,oof_xgb)

    print('MSE:', cross_validation_loss)
    if(output):
        res[1] = predictions_xgb
        mean = res[1].mean()
        print('mean:',mean)
        res.to_csv("./xgb_base.csv", index=False,header = None)
        return res,oof_xgb
    else:
        return predictions_xgb,oof_xgb,cross_validation_loss



def Basyen_stacking(oof_lgb,oof_xgb,predictions_lgb,predictions_xgb,sub,target):
    res = sub.copy()
    # 将lgb和xgb的结果进行stacking
    train_stack = np.vstack([oof_lgb,oof_xgb]).transpose()
    test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

    folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
    oof_stack = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):
        print("fold {}".format(fold_))
        trn_data, trn_y = train_stack[trn_idx], target[trn_idx]
        val_data, val_y = train_stack[val_idx], target[val_idx]
        
        clf_3 = BayesianRidge()
        clf_3.fit(trn_data, trn_y)
        
        oof_stack[val_idx] = clf_3.predict(val_data)
        predictions += clf_3.predict(test_stack) / 10
    
    cross_validation_loss = mean_squared_error(target, oof_stack)
    print(cross_validation_loss)

    res[1] = predictions
    mean = res[1].mean()
    print('mean:',mean)
    res.to_csv("./Basyen_stacking.csv",index=False,header = None)
    return cross_validation_loss
 