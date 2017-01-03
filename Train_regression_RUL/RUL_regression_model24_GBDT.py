# this file is to use a GBDT regressor based on RNN output ogf model 16

import numpy as np
import cPickle as pickle
import time
import csv
from xgboost import XGBRegressor
from sklearn import cross_validation
from matplotlib import pyplot as plt
from sklearn import metrics

# start time
start_time = time.time()

# data file path
data_file_path = '/home/liuyingqi/Desktop'

# model number
model_num = 24
RNN_model_used = 24

test_set = [0,2,5,8,13,17,22,25,28,34]
train_set = [1,3,4,9,10,11,12,15,16,18,19,20,21,23,24,27,30,31,32,33]

train_x = np.empty((0,610),dtype=np.float64)
train_y = np.empty(0,dtype=np.float64)

# generating train set
print ' ------------------ generating trainset ---------------------------'
for File_num in train_set:
    # load RNN out put
    print ' --------------------- loading RNN output ------------------- '
    with open(data_file_path+'/CMAPSS_project/Graphe_result_saved/Model'+str(RNN_model_used)+'/File'+str(File_num+1)+'/RNN_output_RUL.pkl','rb') as file:
        info = pickle.load(file)
        print info.shape
        train_x = np.append(train_x,info,axis=0)

    # load real RUL
    print ' --------------------- loading real RUL ------------------- '
    with open('../Data_strcture.csv') as general_info:
        reader = csv.reader(general_info)
        data_rows = [row for row in reader]
        flight_num = int(data_rows[File_num+1][3])
        print flight_num
        real_RUL = np.arange(flight_num-1,-1,-1)
        print real_RUL.shape
        train_y = np.append(train_y,real_RUL,axis=0)
print train_x.shape
print train_y.shape

test_x = np.empty((0,610),dtype=np.float64)
test_y = np.empty(0,dtype=np.float64)

# generating test set
print ' ------------------ generating testset ---------------------------'
for File_num in test_set:
    # load RNN out put
    print ' --------------------- loading RNN output ------------------- '
    with open(data_file_path+'/CMAPSS_project/Graphe_result_saved/Model'+str(RNN_model_used)+'/File'+str(File_num+1)+'/RNN_output_RUL.pkl','rb') as file:
        info = pickle.load(file)
        print info.shape
        test_x = np.append(test_x,info,axis=0)

    # load real RUL
    print ' --------------------- loading real RUL ------------------- '
    with open('../Data_strcture.csv') as general_info:
        reader = csv.reader(general_info)
        data_rows = [row for row in reader]
        flight_num = int(data_rows[File_num+1][3])
        print flight_num
        real_RUL = np.arange(flight_num-1,-1,-1)
        print real_RUL.shape
        test_y = np.append(test_y,real_RUL,axis=0)
print test_x.shape
print test_y.shape

x_train, x_cv, y_train, y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.01, random_state=42)
regressor = XGBRegressor(max_depth=5,n_estimators=100,reg_alpha=1,reg_lambda=1)
regressor.fit(x_train,y_train)

# print np.mean(test_y - regressor.predict(test_x))
# print metrics.explained_variance_score(test_y,regressor.predict(test_x))
for File_num in test_set:
    # load RNN out put
    print ' --------------------- loading RNN output ------------------- '
    with open(data_file_path+'/CMAPSS_project/Graphe_result_saved/Model'+str(RNN_model_used)+'/File'+str(File_num+1)+'/RNN_output_RUL.pkl','rb') as file:
        info = pickle.load(file)

    # load real RUL
        # load real RUL
    print ' --------------------- loading real RUL ------------------- '
    if File_num + 1 > 10:
        print ' case : no fault '
        with open('../Data_strcture.csv') as general_info:
            reader = csv.reader(general_info)
            data_rows = [row for row in reader]
            flight_num = int(data_rows[File_num + 1][3])
            print 'flight : ' + str(flight_num)
            real_RUL = np.arange(flight_num, 0, -1) - 1
    else:
        print ' case : with fault '
        with open('../Data_strcture.csv') as general_info:
            reader = csv.reader(general_info)
            data_rows = [row for row in reader]
            flight_num = int(data_rows[File_num + 1][3])
            print ' flight : ' + str(flight_num)
            fault_flight = int(data_rows[File_num + 1][4])
            real_RUL = np.append(np.ones(fault_flight - 1) * (flight_num - fault_flight + 1),
                                 np.arange(flight_num - fault_flight + 1, 0, -1) - 1)

    plt.figure()
    plt.plot(real_RUL)
    plt.plot(regressor.predict(info))
    plt.show()