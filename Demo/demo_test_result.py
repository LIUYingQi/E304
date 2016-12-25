# this program is to show a model'result after each trainning step

import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np

# test_set = [1,3,5,7,9,12,17,22,27,32]
# train_set = [0,2,4,6,8,10,11,13,14,15,16,18,19,20,21,23,24,25,26,28,29,30,31,33,34]

# test_set = [1,3,5,9,12,17,22,27,32,34]
# test_set = [0,2,4,8,10,11,13,15,16,18,19,20,21,23,24,25,28,30,31,33]

# test_set = [0,2,5,9,12,17,22,27,32,34]
# train_set = [1,3,4,8,10,11,13,15,16,18,19,20,21,23,24,25,28,30,31,33]

test_set = [0,2,5,8,13,17,22,25,28,34]
train_set = [1,3,4,9,10,11,12,15,16,18,19,20,21,23,24,27,30,31,32,33]

Model_num = 16
param = 'RUL'

# load data

path = '../Test_result_saved/Model'+str(Model_num)+'/mean_diff_'+param+'.pkl'
with open(path,'rb') as test_result:
    mean_diff = pickle.load(test_result)
    print mean_diff

path = '../Test_result_saved/Model'+str(Model_num)+'/var_diff_'+param+'.pkl'
with open(path,'rb') as test_result:
    var_diff = pickle.load(test_result)
    print var_diff

mean_diff = np.array(mean_diff,dtype=np.float32)
var_diff = np.array(var_diff,dtype=np.float32)

steps = np.array(np.arange(mean_diff.size)+1,dtype=np.float32)
print steps
print mean_diff
print var_diff

# plot
fig1 = plt.figure('fig1')
plt.plot(steps,mean_diff)
plt.xlabel('step ')
plt.ylabel('mean difference prediction and real '+param)
plt.title('mean for difference between prediction '+param+' and real '+param+'  for  '+str(Model_num))


fig2 = plt.figure('fig2')
plt.plot(steps,var_diff)
plt.xlabel('step ')
plt.ylabel('var difference prediction and real '+param)
plt.title('variance for difference between prediction '+param+' and real '+param+'  for  '+str(Model_num))
plt.show()