# this file is to show how RNN output changes

# test set :
# test_set = [1,3,5,7,9,12,17,22,27,32]
# train_set = [0,2,4,6,8,10,11,13,14,15,16,18,19,20,21,23,24,25,26,28,29,30,31,33,34]

# test_set = [1,3,5,9,12,17,22,27,32,34]
# test_set = [0,2,4,8,10,11,13,15,16,18,19,20,21,23,24,25,28,30,31,33]

# test_set = [0,2,5,9,12,17,22,27,32,34]
# train_set = [1,3,4,8,10,11,13,15,16,18,19,20,21,23,24,25,28,30,31,33]

test_set = [0,2,5,8,13,17,22,25,28,34]
train_set = [1,3,4,9,10,11,12,15,16,18,19,20,21,23,24,27,30,31,32,33]

titleset = ['alt','MN','TRA','Wf','Fn','SmHPC','SmLPC','SmFan','T48','T2','T24','T30'
    ,'T50','P2','P15','P30','Nf','Nc','epr','phi','Ps30','NfR','NcR','BPR','farB','htBleed','PCNfRdmd','W31','W32'
            ,'forward difference alt','forward difference MN','forward difference TRA','forward difference Wf',
            'forward difference Fn','forward difference SmHPC','forward difference SmLPC','forward difference SmFan',
            'forward difference T48','forward difference T2','forward difference T24','forward difference T30'
    ,'forward difference T50','forward difference P2','forward difference P15','forward difference P30','forward difference Nf','forward difference Nc','forward difference epr',
            'forward difference phi','forward difference Ps30','forward difference NfR','forward difference NcR','forward difference BPR','forward difference farB',
            'forward difference htBleed','forward difference PCNfRdmd','forward difference W31','forward difference W32'
            ,'time','fuel flow','fuel efficiency']

import cPickle as pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

model_num = 16
param = 'RUL'
file = 18
time_step_size = 10
LSTM_size = 61
n_step_feature = 10

for feature in range(LSTM_size):
    plt.figure(figsize=(10, 8))
    plt.subplot(111)
    plt.title('deep feature of last 10 steps output of RNN extracted from signal ' + titleset[feature],fontsize=16)
    plt.xlabel('flight number',fontsize=16)
    plt.ylabel('output values',fontsize=16)
    plt.ylim(-0.5,0.5)
    # plt.axis([0, 300])
    with open('../Graphe_result_saved/Model' + str(model_num) + '/File' + str(file) + '/RNN_output_'+param+'.pkl',
              'rb') as RNN_output:
        RNN_output = np.array(pickle.load(RNN_output))
        RNN_output = np.reshape(RNN_output, (RNN_output.shape[0], time_step_size, LSTM_size))
    for step in range(n_step_feature):
            plt.plot(np.arange(len(RNN_output[:, step, feature])) + 1,RNN_output[:, step, feature])
    plt.show()