
import csv
import pandas as pd
from scipy.stats import norm
from numpy import mean
from numpy import std
from numpy import absolute
import numpy as np
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import scale
from sklearn import metrics
from scipy import spatial
import matplotlib.pyplot as plt



print("ADELE processing has been started, it might take some time")

module_count = {}
cdf_module_count = {}
avg_module_count = {}
cdf_avg_module_count = {}
mean_time = {}
cdf_mean_time = {}
mean_msg = {}
cdf_mean_msg = {}
sev_0 = {}
cdf_sev_0 = {}
sev_1 = {}
cdf_sev_1 = {}
sev_2 = {}
cdf_sev_2 = {}
sev_3 = {}
cdf_sev_3 = {}
sev_4 = {}
cdf_sev_4 = {}
sev_5 = {}
cdf_sev_5 = {}
sev_6 = {}
cdf_sev_6 = {}
sev_7 = {}
cdf_sev_7 = {}
interval_1 = {}
cdf_interval_1 = {}
interval_2 = {}
cdf_interval_2 = {}
interval_3 = {}
cdf_interval_3 = {}
interval_4 = {}
cdf_interval_4 = {}
interval_5 = {}
cdf_interval_5 = {}
interval_6 = {}
cdf_interval_6 = {}




# for i in range(581):
#     module_count[module_name_df.iloc[i][1]] = 0
#     avg_module_count[module_name_df.iloc[i][1]] = 0
#     mean_time[module_name_df.iloc[i][1]] = 0
#     mean_msg[module_name_df.iloc[i][1]] = 0

list = []


k = 3
c = 0
import os
import glob
path = r'700002209254'
csv_files = glob.glob(os.path.join(path, "*.csv"))
for f in csv_files:
    df = pd.read_csv(f)
    c=c+1
    if c<138-k:
        continue
    elif c>167-k and c< 173-k:
        continue
    elif c>182-k:
        continue
    for j in range(331):
        if c == 138-k:
            module_count[df.iloc[j,0]] = [df.iloc[j,2]]
            avg_module_count[df.iloc[j,0]] = [df.iloc[j,3]]
            mean_time[df.iloc[j,0]] = [df.iloc[j,4]]
            mean_msg[df.iloc[j,0]] = [df.iloc[j,5]]
            sev_0[df.iloc[j,0]] = [df.iloc[j,6]]
            sev_1[df.iloc[j,0]] = [df.iloc[j,7]]
            sev_2[df.iloc[j,0]] = [df.iloc[j,8]]
            sev_3[df.iloc[j,0]] = [df.iloc[j,9]]
            sev_4[df.iloc[j,0]] = [df.iloc[j,10]]
            sev_5[df.iloc[j,0]] = [df.iloc[j,11]]
            sev_6[df.iloc[j,0]] = [df.iloc[j,12]]
            sev_7[df.iloc[j,0]] = [df.iloc[j,13]]
            interval_1[df.iloc[j,0]] = [df.iloc[j,14]]
            interval_2[df.iloc[j,0]] = [df.iloc[j,15]]
            interval_3[df.iloc[j,0]] = [df.iloc[j,16]]
            interval_4[df.iloc[j,0]] = [df.iloc[j,17]]
            interval_5[df.iloc[j,0]] = [df.iloc[j,18]]
            interval_6[df.iloc[j,0]] = [df.iloc[j,19]]
            
        else :
            module_count[df.iloc[j,0]].append(df.iloc[j,2])
            avg_module_count[df.iloc[j,0]].append(df.iloc[j,3])
            mean_time[df.iloc[j,0]].append(df.iloc[j,4])
            mean_msg[df.iloc[j,0]].append(df.iloc[j,5])
            sev_0[df.iloc[j,0]].append(df.iloc[j,6])
            sev_1[df.iloc[j,0]].append(df.iloc[j,7])
            sev_2[df.iloc[j,0]].append(df.iloc[j,8])
            sev_3[df.iloc[j,0]].append(df.iloc[j,9])
            sev_4[df.iloc[j,0]].append(df.iloc[j,10])
            sev_5[df.iloc[j,0]].append(df.iloc[j,11])
            sev_6[df.iloc[j,0]].append(df.iloc[j,12])
            sev_7[df.iloc[j,0]].append(df.iloc[j,13])
            interval_1[df.iloc[j,0]].append(df.iloc[j,14])
            interval_2[df.iloc[j,0]].append(df.iloc[j,15])
            interval_3[df.iloc[j,0]].append(df.iloc[j,16])
            interval_4[df.iloc[j,0]].append(df.iloc[j,17])
            interval_5[df.iloc[j,0]].append(df.iloc[j,18])
            interval_6[df.iloc[j,0]].append(df.iloc[j,19])

           
    # for j in range(581):
        
    #     module_count[j] += df.iloc[j,1]
    #     avg_module_count[j] += df.iloc[j,2]
    #     mean_time[j] += df.iloc[j,3]
    #     mean_msg[j] += df.iloc[j,4]

print("cases reading done")

for key in range(331):
    cdf_module_count[key] = norm.cdf(module_count[key])
    cdf_avg_module_count[key] = norm.cdf(avg_module_count[key])
    cdf_mean_time[key] = norm.cdf(mean_time[key])
    cdf_mean_msg[key] = norm.cdf(mean_msg[key])
    cdf_sev_0[key] =  norm.cdf(sev_0[key])
    cdf_sev_1[key] = norm.cdf(sev_1[key])
    cdf_sev_2[key] = norm.cdf(sev_2[key])
    cdf_sev_3[key] = norm.cdf(sev_3[key])
    cdf_sev_4[key] = norm.cdf(sev_4[key])
    cdf_sev_5[key] = norm.cdf(sev_5[key])
    cdf_sev_6[key] = norm.cdf(sev_6[key])
    cdf_sev_7[key] = norm.cdf(sev_7[key])
    cdf_interval_1[key] = norm.cdf(interval_1[key])
    cdf_interval_2[key] = norm.cdf(interval_2[key])
    cdf_interval_3[key] = norm.cdf(interval_3[key])
    cdf_interval_4[key] = norm.cdf(interval_4[key])
    cdf_interval_5[key] = norm.cdf(interval_5[key])
    cdf_interval_6[key] = norm.cdf(interval_6[key])
total = 0

print("Interval attributes calculated")

ridge = Ridge()
coefs = []

# module = []

# for i in range(581):
#     for j in n:
#         df = pd.read_csv(r'score_matrix_day_wise/day_{}.csv'.format(j))
#         list = df['module_count'].to_list()
#         module.append(df[i])
    

frames = []


i=0
c=0
for f in csv_files:
    
    df = pd.read_csv(f)
    c =c+1
    if c<138-k:
        continue
    elif c>167-k and c< 173-k:
        continue
    elif c>182-k:
        continue
    i=i+1
    for j in range(331):
        df.iloc[j,2] =  (2*abs(0.5 - cdf_module_count[j][i-1]))
        df.iloc[j,3] = (2*abs(0.5 - cdf_avg_module_count[j][i-1]))
        df.iloc[j,4] = (2*abs(0.5 - cdf_mean_time[j][i-1]))
        df.iloc[j,5] = (2*abs(0.5 - cdf_mean_msg[j][i-1]))
        df.iloc[j,6] = (2*abs(0.5 - cdf_sev_0[j][i-1]))
        df.iloc[j,7] = (2*abs(0.5 - cdf_sev_1[j][i-1]))
        df.iloc[j,8] = (2*abs(0.5 - cdf_sev_2[j][i-1]))
        df.iloc[j,9] = (2*abs(0.5 - cdf_sev_3[j][i-1]))
        df.iloc[j,10] = (2*abs(0.5 - cdf_sev_4[j][i-1]))
        df.iloc[j,11] = (2*abs(0.5 - cdf_sev_5[j][i-1]))
        df.iloc[j,12] = (2*abs(0.5 - cdf_sev_6[j][i-1]))
        df.iloc[j,13] = (2*abs(0.5 - cdf_sev_7[j][i-1]))
        df.iloc[j,14] = (2*abs(0.5 - cdf_interval_1[j][i-1]))
        df.iloc[j,15] = (2*abs(0.5 - cdf_interval_2[j][i-1]))
        df.iloc[j,16] = (2*abs(0.5 - cdf_interval_3[j][i-1]))
        df.iloc[j,17] = (2*abs(0.5 - cdf_interval_4[j][i-1]))
        df.iloc[j,18] = (2*abs(0.5 - cdf_interval_5[j][i-1]))
        df.iloc[j,19] = (2*abs(0.5 - cdf_interval_6[j][i-1]))
        
    frames.append(df)
    
print("dataframes processed")
    
    #print(df.columns[[0]])
    # df.drop(df.columns[[0]], axis=1, inplace=True)
    # X, y = df[df.columns.difference(["severity"])], df[['severity']]
    # # print(y)
    # ridge.set_params(alpha=0.5)
    # ridge.fit(X,y)
    # for i in n:

    # print(ridge.coef_)
    # coefs.append(np.sum(ridge.coef_))
    # model = Ridge(alpha=0.5)
    # cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    # scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # scores = absolute(scores)
    # print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
    # total += mean(scores)



arr = []
for t in range(40):
    if t>29:
        arr.append(1)
    else :
        arr.append(0)

truth = []

for t in range(30):
    if t>19:
        truth.append(1)
    else :
        truth.append(0)

frames = np.array(frames)

# matrix.drop(matrix.columns[[0]], axis=1, inplace=True)
X, y = frames[:,:,2:], arr
# print(X)
# print(X[np.nonzero(X)])
y = np.array(y)
X = X.reshape(40,18*331)
ridge.set_params(alpha=0.5)
ridge.fit(X,y)

print("regression fitted")

i=0
c=0
pred = []
test = []
res = []
Fatal_score = []
Nrmal_score = []
score = []
tp = 0
fn = 0
tn = 0
fp = 0
for f in csv_files:
    
    df = pd.read_csv(f)
    c =c+1
    if c<98-k:
        continue
    elif c>117-k and c< 168-k:
        continue
    elif c >173-k and c<184-k:
        continue
    i=i+1
    for j in range(331):
        df.iloc[j,2] =  (2*abs(0.5 - cdf_module_count[j][i-1]))
        df.iloc[j,3] = (2*abs(0.5 - cdf_avg_module_count[j][i-1]))
        df.iloc[j,4] = (2*abs(0.5 - cdf_mean_time[j][i-1]))
        df.iloc[j,5] = (2*abs(0.5 - cdf_mean_msg[j][i-1]))
        df.iloc[j,6] = (2*abs(0.5 - cdf_sev_0[j][i-1]))
        df.iloc[j,7] = (2*abs(0.5 - cdf_sev_1[j][i-1]))
        df.iloc[j,8] = (2*abs(0.5 - cdf_sev_2[j][i-1]))
        df.iloc[j,9] = (2*abs(0.5 - cdf_sev_3[j][i-1]))
        df.iloc[j,10] = (2*abs(0.5 - cdf_sev_4[j][i-1]))
        df.iloc[j,11] = (2*abs(0.5 - cdf_sev_5[j][i-1]))
        df.iloc[j,12] = (2*abs(0.5 - cdf_sev_6[j][i-1]))
        df.iloc[j,13] = (2*abs(0.5 - cdf_sev_7[j][i-1]))
        df.iloc[j,14] = (2*abs(0.5 - cdf_interval_1[j][i-1]))
        df.iloc[j,15] = (2*abs(0.5 - cdf_interval_2[j][i-1]))
        df.iloc[j,16] = (2*abs(0.5 - cdf_interval_3[j][i-1]))
        df.iloc[j,17] = (2*abs(0.5 - cdf_interval_4[j][i-1]))
        df.iloc[j,18] = (2*abs(0.5 - cdf_interval_5[j][i-1]))
        df.iloc[j,19] = (2*abs(0.5 - cdf_interval_6[j][i-1]))
    test =df.to_numpy()    
    l = test[:,2:]
    l = l.reshape(-1,18*331)
    pred = ridge.predict(l)
    if i>19:
        Fatal_score.append(pred)
    else :
        Nrmal_score.append(pred)
    score.append(pred)
    if pred>0.2:
        res.append(1)
    else:
        res.append(0)
    # res.append(pred)
    # print(pred)
  
nrml = float(sum(Nrmal_score))/float(len(Nrmal_score))
fatal = float(sum(Fatal_score))/float(len(Fatal_score))

new_acc =[]

for i in range(len(Nrmal_score)):
    if abs(nrml-Nrmal_score[i])< abs(fatal-Nrmal_score[i]):
        new_acc.append(1)
    else:
        new_acc.append(0)

for i in range(len(Fatal_score)):
    if abs(fatal-Fatal_score[i])< abs(nrml-Fatal_score[i]):
        new_acc.append(1)
    else:
        new_acc.append(0)    


# def get_bin(x):
#     if x<0.2:
#         return 0
#     else:
#         return 1

def get_bin_acc(test, pred):
    a=[]
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for i in range(0,len(test)):
        if (test[i])==(pred[i]):
            a.append(1)
            if test[i] == 1:
                tp +=1
            else:
                tn +=1
        else:
            if test[i] == 1:
                fn +=  1
            else:
                fp +=1
            a.append(0)
    return a,tp,tn,fn,fp		

acc,tp,tn,fn,fp = get_bin_acc(truth, res)
# acc=(1-spatial.distance.cosine(arr, score))
# # print(acc)
print("ACC ",float(sum(new_acc))/float(len(new_acc)))
print("stopped")
# print(tp,fn,tn,fp)    
# print("True Positive Rate",float(tp)/(float(tp)+(float(fn)/5)))
# print("False Positive Rate",float(fp)/(float(fp)+(float(tn)/5)))

# acc = metrics.accuracy_score(np.array(arr), np.array(ridge.coef_))
# results = 0
# for i in n :
#     df = pd.read_csv(r'feature_matrix_day_wise/day_{}.csv'.format(i))
#     for j in range(581):
#         df.iloc[j,1] =  (2*abs(0.5 - cdf_module_count[j][i-1]))
#         df.iloc[j,2] = (2*abs(0.5 - cdf_avg_module_count[j][i-1]))
#         df.iloc[j,3] = (2*abs(0.5 - cdf_mean_time[j][i-1]))
#         df.iloc[j,4] = (2*abs(0.5 - cdf_mean_msg[j][i-1]))
#         if df.iloc[j,5] == 'FATAL' : 
#             df.iloc[j,5] = float(1)
#         elif df.iloc[j,5] == 'WARNING':
#             df.iloc[j,5] = float(0.5)
#         else :
#             df.iloc[j,5] = float(0)
#         df.iloc[j,6] = (2*abs(0.5 - cdf_interval_1[j][i-1]))
#         df.iloc[j,7] = (2*abs(0.5 - cdf_interval_2[j][i-1]))
#         df.iloc[j,8] = (2*abs(0.5 - cdf_interval_3[j][i-1]))
#         df.iloc[j,9] = (2*abs(0.5 - cdf_interval_4[j][i-1]))
#         df.iloc[j,10] = (2*abs(0.5 - cdf_interval_5[j][i-1]))
#         df.iloc[j,11] = (2*abs(0.5 - cdf_interval_6[j][i-1]))
#     df.drop(df.columns[[0]], axis=1, inplace=True)
#     pred = ridge.predict(df[df.columns.difference(["severity"])])
#     tr = df[['severity']].to_numpy()
#     # print(type(pred))
#     # print(type(tr))
#     tolerance = 1e-50
#     results += (np.abs(pred - tr) < tolerance ).all().mean()
#     print(results)
# for i in range(581):
#     module_count[module_name_df.iloc[i][1]] = module_count[module_name_df.iloc[i][1]]/30
#     avg_module_count[module_name_df.iloc[i][1]] = avg_module_count[module_name_df.iloc[i][1]]/30
#     mean_time[module_name_df.iloc[i][1]] = mean_time[module_name_df.iloc[i][1]]/30
#     mean_msg[module_name_df.iloc[i][1]] = mean_msg[module_name_df.iloc[i][1]]/30