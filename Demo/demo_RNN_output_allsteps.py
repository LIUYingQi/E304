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
            ,'alt','MN','TRA','Wf','Fn','SmHPC','SmLPC','SmFan','T48','T2','T24','T30'
    ,'T50','P2','P15','P30','Nf','Nc','epr','phi','Ps30','NfR','NcR','BPR','farB','htBleed','PCNfRdmd','W31','W32'
            ,'time','fuel flow','fuel efficiency']

import cPickle as pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

model_num = 16
param = 'RUL'
fault_file_1 = 1
time_step_size = 10
LSTM_size = 61
n_step_feature = 3


with open('../Data_strcture.csv','rb') as Data_structure:
    info = pd.read_csv(Data_structure)
    print info
    fault_flight = info['Fault_flight'][fault_file_1]
    fault_flight_1 = int(fault_flight)
    print fault_flight_1
    #
    # norme_flight = info['Fault_flight'][normal_file]
    # norme_flight = int(norme_flight)
    # print norme_flight

for feature in range(LSTM_size):
    plt.figure(figsize=(10, 8))
    plt.subplot(111)
    plt.title(titleset[feature])
    # plt.axis([0, 300])
    with open('../Graphe_result_saved/Model' + str(model_num) + '/File' + str(1) + '/RNN_output_'+param+'.pkl',
              'rb') as RNN_RUL:
        RNN_RUL = np.array(pickle.load(RNN_RUL))
        RNN_RUL = np.reshape(RNN_RUL, (RNN_RUL.shape[0], time_step_size, LSTM_size))
    for step in range(n_step_feature):
            plt.plot(np.arange(len(RNN_RUL[:, step, feature])) + 1,RNN_RUL[:, step, feature])
    plt.show()