import pandas as pd
import numpy as np
import lightgbm as lgb
import time,datetime
import matplotlib.pyplot as plt
import gensim
from util import *
from gensim.models import word2vec
from scipy import sparse
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold
import sklearn,gc,re
import seaborn as sns
from sklearn.linear_model import LinearRegression,BayesianRidge,LinearRegression
import warnings
warnings.filterwarnings('ignore')

path = '.'
train1 = pd.read_csv('./data/jinnan_round1_train_20181227.csv', encoding = 'gb18030')
train2  = pd.read_csv('./data/jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
train3  = pd.read_csv('./data/jinnan_round1_testB_20190121.csv', encoding = 'gb18030')
train4  = pd.read_csv('./data/jinnan_round1_test_20190201.csv', encoding = 'gb18030')
ans2 = pd.read_csv('./data/jinnan_round1_ansA_20190125.csv', encoding = 'gb18030',header = None)
ans3 = pd.read_csv('./data/jinnan_round1_ansB_20190125.csv', encoding = 'gb18030',header = None)
ans4 = pd.read_csv('./data/jinnan_round1_ans_20190201.csv', encoding = 'gb18030',header = None)
train2['收率'] = ans2[1].values
train3['收率'] = ans3[1].values
train4['收率'] = ans4[1].values
train = pd.concat([train1,train2,train3,train4],axis = 0)
train = train[train['收率'] >= 0.85]
train = train[train['收率'] <= 1]

test = pd.read_csv('./data/FuSai.csv', encoding = 'gb18030')
#test2 = pd.read_csv('./data/FuSai.csv', encoding = 'gb18030')
sub  = test[['样本id']]
###These feature's number of unique is one value in test dataset.
# 删除类别唯一的特征
target = train['收率']
for df in [train, test]:
    df.drop(['B3', 'B13','A13', 'A18', 'A23'], axis=1, inplace=True)


# 合并数据集
del train['收率']
data = pd.concat([train,test],axis=0,ignore_index=True)
data = data.fillna(-1)
del data['样本id']
data.loc[data['B14'] == 785,'B14'] = 385.0
#data = data.sort_values(['样本id'],ascending = True)

###处理时间数据和部分异常数据
for f in ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']:
    data[f + '_starttime'] = data[f].apply(timestart1)
    data[f] = data[f].apply(timeTranSecond)

for f in ['A20','A28','B4','B9','B10','B11']:
    data[f + '_starttime'] = data[f].apply(timestart2)
    data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)
data['A25'] = data['A25'].apply(lambda x:int(x) if x!= '1900/3/10 0:00' else -1)
old_data = data.copy()

###1.onehot特征，包括时间
onehot_list = [i for i in data.columns]
onehot_data = encode_onehot(data,onehot_list)

###2.count和ratio特征：
count_feature_list = []
def feature_count(data, features=[], is_feature=True):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    nunique = []
    for i in features:
        nunique.append(data[i].nunique())
        new_feature += '_' + i
    if len(features) > 1 and len(data[features].drop_duplicates()) <= np.max(nunique):
        print(new_feature, 'is unvalid cross feature:')
        return data
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
    data = data.merge(temp, how = 'left', on=features)
    if is_feature:
        count_feature_list.append(new_feature)
    return data

def get_count(data):
    ###第三部分 feature count就是统计每个特征的size
    count_list = [i for i in data.columns if i!='样本id']
    for i in count_list:
        n = data[i].nunique()
        if n > 5:
            data = feature_count(data, [i])
    return data

count_data = get_count(data)[count_feature_list]

#####3.cross特征

def get_cross_feature(new_data):
    data = new_data.copy()
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
          
    for i in A_features:
        for j in ['B14']:
            col_name = "cross_" + i + "_and_" + j
            cross_features.append(col_name)
            data[col_name] = data[i].astype(str).values + '_' + data[j].astype(str).values
           
    
    for f in cross_features:
        data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
    
    data = data[cross_features]
    return data

cross_data = get_cross_feature(old_data)

####4.对B14进行分箱操作：
#$##直接cut没用，最后预测结果一样
def get_numsplit(new_data):
    data = new_data.copy()
    data['cut_B14'] = pd.cut(data['B14'], 5, labels=False)
    data = pd.get_dummies(data, columns=['cut_B14'])
    mean_features = []
    li = ['cut_B14_0','cut_B14_2','cut_B14_3','cut_B14_4']
    for f1 in [i for i in data.columns if ('B' in i) and ('cut' not in i)]:
        for f2 in li:
            col_name = f1+"_"+f2+'_mean'
            mean_features.append(col_name)
            order_label = data.groupby([f1])[f2].mean()
            data[col_name] = data[f1].map(order_label)
    data = data[mean_features]
    return data
B14_split = get_numsplit(old_data)


def get_add_divide_features(data):
    new_data = data.copy()
    drop_col = data.columns
    new_data['b14/a1_a3_a4_a19_b1_b12'] = new_data['B14']/(new_data['A1']+new_data['A3']+new_data['A4']+new_data['A19']+data['B1']+new_data['B12'])
    new_data['tem_A8_A6_diff'] = new_data['A8'] - new_data['A6'] 
    new_data['A7_A5_diff'] = new_data['A7'] - new_data['A5']
    new_data['A7_A5_diff'] = new_data['A7_A5_diff'].apply(lambda x: x if x>0 else x+24)
    new_data['tem_A8_A6_rate'] = new_data['tem_A8_A6_diff']/new_data['A7_A5_diff']
        
    new_data['tem_A10_A8_diff'] = new_data['A10'] - new_data['A8'] 
    new_data['A9_A7_diff'] = new_data['A9'] - new_data['A7']
    new_data['A9_A7_diff'] = new_data['A9_A7_diff'].apply(lambda x: x if x>0 else x+24)
    new_data['tem_A10_A8_diff'] = new_data['tem_A10_A8_diff']/new_data['A9_A7_diff']
    
    new_data['tem_A12_A10_diff'] = new_data['A12'] - new_data['A10'] 
    new_data['A11_A9_diff'] = new_data['A11'] - new_data['A9']
    new_data['A11_A9_diff'] = new_data['A11_A9_diff'].apply(lambda x: x if x>0 else x+24)
    new_data['tem_A12_A10_diff'] = new_data['tem_A12_A10_diff']/new_data['A11_A9_diff']
        
    
    new_data['tem_A15_A12_diff'] = new_data['A15'] - new_data['A12'] 
    new_data['A14_A11_diff'] = new_data['A14'] - new_data['A11']
    new_data['A14_A11_diff'] = new_data['A14_A11_diff'].apply(lambda x: x if x>0 else x+24)
    new_data['tem_A15_A12_diff'] = new_data['tem_A15_A12_diff']/new_data['A14_A11_diff']
        
    
    new_data['tem_A17_A15_diff'] = new_data['A17'] - new_data['A15'] 
    new_data['A16_A14_diff'] = new_data['A16'] - new_data['A14']
    new_data['A16_A14_diff'] = new_data['A16_A14_diff'].apply(lambda x: x if x>0 else x+24)
    new_data['tem_A17_A15_diff'] = new_data['tem_A17_A15_diff']/new_data['A16_A14_diff']
        
    
    new_data['tem_A27_A25_diff'] = new_data['A27'] - new_data['A25'] 
    new_data['A26_A24_diff'] = new_data['A26'] - new_data['A24']
    new_data['A26_A24_diff'] = new_data['A26_A24_diff'].apply(lambda x: x if x>0 else x+24)
    new_data['tem_A27_A25_diff'] = new_data['tem_A27_A25_diff']/new_data['A26_A24_diff']
        
    
    new_data['A_temp_sum'] = new_data['A6'] + new_data['A8'] + new_data['A10'] + new_data['A12'] + new_data['A15'] + new_data['A17'] + new_data['A25'] + new_data['A27']
    new_data['A1_temp_sum'] = new_data['A6'] + new_data['A8'] + new_data['A10'] + new_data['A12'] + new_data['A15'] + new_data['A17']
    new_data['A2_temp_sum'] = new_data['A25'] + new_data['A27']
    new_data['A_material_sum'] = new_data['A1'] + new_data['A2'] + new_data['A3'] + new_data['A4'] + new_data['A19'] + new_data['A21'] + new_data['A22'] 
    new_data['A1_material_sum'] = new_data['A1'] + new_data['A2'] + new_data['A3'] + new_data['A4'] + new_data['A19']
    new_data['A2_material_sum'] = new_data['A21'] + new_data['A22'] 
    new_data['B_material_sum'] = new_data['B1'] + new_data['B2'] + new_data['B12'] + new_data['B14']
    new_data['material_sum'] = new_data['A_material_sum'] + new_data['B_material_sum']
    for i in ['A1','A2','A3','A4','A19','A21','A22','B1','B2','B12','B14']:
        new_data['%s_div_material_sum'%i] = new_data[i] / new_data['material_sum']
    for i in ['A1','A2','A3','A4','A19','A21','A22','B1','B2','B12']:
        new_data['%s_div_B14'%i] = new_data[i] / new_data['B14']
        new_data['B14_div_%s'%i] = new_data['B14'] / new_data[i]
    ###在xgb 12354的基础上加
    for i in ['A1_material_sum','A2_material_sum','material_sum']:
        for j in ['A1','A2','A3','A4','A19','A21','A22','B1','B2','B12','B14']:
            new_data['%s_div_%s'%(j,i)] = new_data[j] / new_data[i]
    for i in ['B_material_sum']:
        for j in ['B1','B2','B12','B14']:
            new_data['%s_div_%s'%(j,i)] = new_data[j] / new_data[i]
    new_data = new_data.drop(drop_col,axis = 1)
    return new_data


###5.做了一些div特征，线下从0.13976-0.12822
add_div_fea = get_add_divide_features(old_data)

###7. 连续差值特征
#diff_features = get_diff(old_data)

drop_col = ['A1','A2','A3','A16']
new_list = [i for i in old_data.columns if i not in drop_col]
new_old_data = old_data[new_list]

#new_old_data = pd.merge(new_old_data,diff_features,on = ['样本id'],how = 'left')
final_data = pd.concat([new_old_data,cross_data,B14_split,add_div_fea],axis = 1)
print(final_data.shape)
train_data = final_data[:train.shape[0]]
test_data = final_data[train.shape[0]:]

result_lgb,oof_lgb,lgb_loss = baseline_para_lgb(train_data,target.values,test_data,sub,output = False)
result_xgb,oof_xgb,xgb_loss = baseline_para_xgb(train_data.values,target.values,test_data.values,sub,output = False)

stacking_loss = Basyen_stacking(oof_lgb,oof_xgb,result_lgb,result_xgb,sub,target.values)
loss = {'lgb_loss':lgb_loss,'xgb_loss':xgb_loss,'stacking_loss':stacking_loss}
for i,j in loss.items():
	print(i+':{}'.format(j))