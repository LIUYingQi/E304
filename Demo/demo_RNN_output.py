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
fault_file_1 =5
normal_file_1 = 22
fault_file_2 = 8
normal_file_2 = 25
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


with open('../Graphe_result_saved/Model' + str(model_num) + '/File' + str(fault_file_1 + 1) + '/RNN_output_RUL.pkl','rb') as RNN_fault_RUL_1,\
        open('../Graphe_result_saved/Model' + str(model_num) + '/File' + str(normal_file_1 + 1) + '/RNN_output_RUL.pkl','rb') as RNN_normal_RUL_1,\
        open('../Graphe_result_saved/Model' + str(model_num) + '/File' + str(fault_file_1 + 1) + '/RNN_output_HI.pkl','rb') as RNN_fault_HI_1,\
        open('../Graphe_result_saved/Model' + str(model_num) + '/File' + str(normal_file_1 + 1) + '/RNN_output_HI.pkl','rb') as RNN_normal_HI_1,\
        open('../Graphe_result_saved/Model' + str(model_num) + '/File' + str(fault_file_2 + 1) + '/RNN_output_RUL.pkl','rb') as RNN_fault_RUL_2,\
        open('../Graphe_result_saved/Model' + str(model_num) + '/File' + str(normal_file_2 + 1) + '/RNN_output_RUL.pkl','rb') as RNN_normal_RUL_2,\
        open('../Graphe_result_saved/Model' + str(model_num) + '/File' + str(fault_file_2 + 1) + '/RNN_output_HI.pkl','rb') as RNN_fault_HI_2,\
        open('../Graphe_result_saved/Model' + str(model_num) + '/File' + str(normal_file_2 + 1) + '/RNN_output_HI.pkl','rb') as RNN_normal_HI_2:

    RNN_output_fault_RUL_1 = np.array(pickle.load(RNN_fault_RUL_1))
    RNN_output_normal_RUL_1 = np.array(pickle.load(RNN_normal_RUL_1))
    RNN_output_fault_HI_1 = np.array(pickle.load(RNN_fault_HI_1))
    RNN_output_normal_HI_1 = np.array(pickle.load(RNN_normal_HI_1))
    RNN_output_fault_RUL_2 = np.array(pickle.load(RNN_fault_RUL_2))
    RNN_output_normal_RUL_2 = np.array(pickle.load(RNN_normal_RUL_2))
    RNN_output_fault_HI_2 = np.array(pickle.load(RNN_fault_HI_2))
    RNN_output_normal_HI_2 = np.array(pickle.load(RNN_normal_HI_2))

    print RNN_output_fault_RUL_1.shape
    RNN_output_fault_RUL_1 = np.reshape(RNN_output_fault_RUL_1,(RNN_output_fault_RUL_1.shape[0],time_step_size,LSTM_size))
    print RNN_output_fault_RUL_1.shape

    print RNN_output_normal_RUL_1.shape
    RNN_output_normal_RUL_1 = np.reshape(RNN_output_normal_RUL_1,(RNN_output_normal_RUL_1.shape[0],time_step_size,LSTM_size))
    print RNN_output_normal_RUL_1.shape

    print RNN_output_fault_HI_1.shape
    RNN_output_fault_HI_1 = np.reshape(RNN_output_fault_HI_1,(RNN_output_fault_HI_1.shape[0],time_step_size,LSTM_size))
    print RNN_output_fault_HI_1.shape

    print RNN_output_normal_HI_1.shape
    RNN_output_normal_HI_1 = np.reshape(RNN_output_normal_HI_1,(RNN_output_normal_HI_1.shape[0],time_step_size,LSTM_size))
    print RNN_output_normal_HI_1.shape

    print RNN_output_fault_RUL_2.shape
    RNN_output_fault_RUL_2 = np.reshape(RNN_output_fault_RUL_2,(RNN_output_fault_RUL_2.shape[0],time_step_size,LSTM_size))
    print RNN_output_fault_RUL_2.shape

    print RNN_output_normal_RUL_2.shape
    RNN_output_normal_RUL_2 = np.reshape(RNN_output_normal_RUL_2,(RNN_output_normal_RUL_2.shape[0],time_step_size,LSTM_size))
    print RNN_output_normal_RUL_2.shape

    print RNN_output_fault_HI_2.shape
    RNN_output_fault_HI_2 = np.reshape(RNN_output_fault_HI_2,(RNN_output_fault_HI_2.shape[0],time_step_size,LSTM_size))
    print RNN_output_fault_HI_2.shape

    print RNN_output_normal_HI_2.shape
    RNN_output_normal_HI_2 = np.reshape(RNN_output_normal_HI_2,(RNN_output_normal_HI_2.shape[0],time_step_size,LSTM_size))
    print RNN_output_normal_HI_2.shape

    for feature in range(LSTM_size):
        # print RNN_output_fault[:,n_step_feature,feature]
        print feature
        plt.figure(figsize=(20,10))
        plt.title(titleset[feature])
        plt.subplot(221)
        plt.title(titleset[feature])
        plt.axis([0,300,-1,1])
        plt.plot(np.arange(len(RNN_output_fault_RUL_1[:,n_step_feature,feature]))+1,RNN_output_fault_RUL_1[:,n_step_feature,feature],color='red')
        plt.plot(np.arange(len(RNN_output_normal_RUL_1[:,n_step_feature,feature]))+1,RNN_output_normal_RUL_1[:,n_step_feature,feature],color='blue')
        plt.legend(['RUL RNN output fault happened','RUL RNN output fault not happened'])
        plt.subplot(222)
        plt.title(titleset[feature])
        plt.axis([0,300,-1,1])
        plt.plot(np.arange(len(RNN_output_fault_HI_1[:, n_step_feature, feature])) + 1,RNN_output_fault_HI_1[:, n_step_feature, feature], color='red')
        plt.plot(np.arange(len(RNN_output_normal_HI_1[:, n_step_feature, feature])) + 1,RNN_output_normal_HI_1[:, n_step_feature, feature], color='blue')
        plt.legend(["HI RNN output fault happened","HI RNN output not happened"])
        plt.subplot(223)
        plt.title(titleset[feature])
        plt.axis([0,300,-1,1])
        plt.plot(np.arange(len(RNN_output_fault_RUL_2[:,n_step_feature,feature]))+1,RNN_output_fault_RUL_2[:,n_step_feature,feature],color='red')
        plt.plot(np.arange(len(RNN_output_normal_RUL_2[:,n_step_feature,feature]))+1,RNN_output_normal_RUL_2[:,n_step_feature,feature],color='blue')
        plt.legend(['RUL RNN output fault happened','RUL RNN output fault not happened'])
        plt.subplot(224)
        plt.title(titleset[feature])
        plt.axis([0,300,-1,1])
        plt.plot(np.arange(len(RNN_output_fault_HI_2[:, n_step_feature, feature])) + 1,RNN_output_fault_HI_2[:, n_step_feature, feature], color='red')
        plt.plot(np.arange(len(RNN_output_normal_HI_2[:, n_step_feature, feature])) + 1,RNN_output_normal_HI_2[:, n_step_feature, feature], color='blue')
        plt.legend(["HI RNN output fault happened","HI RNN output not happened"])
        plt.show()

    # plt.plot(np.arange(len(RNN_output_normal[-1, :, 58])) + 1,
    #          RNN_output_normal[-1, :, 58], color='blue')
    # plt.show()