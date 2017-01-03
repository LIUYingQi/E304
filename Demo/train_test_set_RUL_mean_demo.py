import numpy as np
import pandas as pd

info = pd.read_csv('../Data_strcture.csv',engine='c')
info['RUL_total'] = info['Flight_num']*(info['Flight_num']+1)/2
# print info

# test_set = [1,3,5,7,9,12,17,22,27,32]
# train_set = [0,2,4,6,8,10,11,13,14,15,16,18,19,20,21,23,24,25,26,28,29,30,31,33,34]

# test_set = [1,3,5,9,12,17,22,27,32,34]
# test_set = [0,2,4,8,10,11,13,15,16,18,19,20,21,23,24,25,28,30,31,33]

# test_set = [0,2,5,9,12,17,22,27,32,34]
# train_set = [1,3,4,8,10,11,13,15,16,18,19,20,21,23,24,25,28,30,31,33]

test_set = [0,2,5,8,13,17,22,25,28,34]
train_set = [1,3,4,9,10,11,12,15,16,18,19,20,21,23,24,27,30,31,32,33]

print 'test set RUL mean:'
print info.iloc[test_set,:]['RUL_total'].sum()/len(test_set)

print 'train set RUL mean:'
print info.iloc[train_set,:]['RUL_total'].sum()/len(train_set)
#
# print 'relation test_set/train_set:'
# print (info.iloc[test_set,:]['RUL_total'].sum()/len(test_set))/(info.iloc[train_set,:]['RUL_total'].sum()/len(train_set))

print 'test set RUL mean:'
print (info.iloc[test_set,:]['RUL_total']/info.iloc[test_set,:]['Flight_num']).sum()/len(test_set)

print 'train set RUL mean:'
print (info.iloc[train_set,:]['RUL_total']/info.iloc[train_set,:]['Flight_num']).sum()/len(train_set)