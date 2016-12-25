# this file is to do classification with random forest

import cPickle as pickle
import numpy as np
import pandas as pd
import csv

model_num = 6
time_step_size = 10
LSTM_size = 61


# pre-define value
test_set_volume = 1000
train_set_volume = 5000

# test set :
train_set = [0,2,4,6,8,10,11,12,13,15,16,17,18,20,21,22,23,25,26,27,28,30,31,32,33]

# test set :
test_set = [1,3,5,8,10,12,16,23,28,34]
probability_set = [0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1, 0.1]

# data file path
data_file_path = '/home/liuyingqi/Desktop'

# load general info from Data_strcture.csv file
with open('../Data_strcture.csv') as general_info:
    reader = csv.reader(general_info)
    data_rows = [ row for row in reader]
    data_rows = data_rows[1:]

# random generating testset
for test_num in range(test_set_volume):

    # randomly choose from 35 engine sample file
    file = np.random.choice(test_set, p=probability_set)
    print 'file'+str(file+1)
    if file < 10:
        engine = (file % 2) + 1
    else:
        engine = ((file - 10) % 5) + 1
    flight_min = 1
    flight_max = data_rows[file][3]
    flight_max = int(flight_max)
    flight = np.random.random_integers(flight_min, flight_max)
    print 'flight:'+str(flight)

    with open('../Graphe_result_saved/Model' + str(model_num) + '/File' + str(file + 1) + '/RNN_output.pkl','rb') as RNN_output_file:
        info = pickle.load(RNN_output_file)
        info = np.reshape(info[flight-1,:],(time_step_size,LSTM_size))
        with open('../Fault_classification_testset/testset'+str(test_num+1)+'.pkl','wb') as to_save:
            # read label fault_type
            file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(engine) + '/EngineFault_type.csv'
        with open(file_name, 'rb') as file_name:
            fault_type_reader = csv.reader(file_name)
            fault_type = [row for row in fault_type_reader]
            fault_type = fault_type[flight][0]

        # read label health indicator
file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(
    engine) + '/EngineHealth.csv'
with open(file_name, 'rb') as file_name:
    engine_health_reader = csv.reader(file_name)
    engine_health = [row for row in engine_health_reader]
    engine_health = engine_health[flight][0]

# read label RUL
file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(
    engine) + '/EngineRUL.csv'
with open(file_name, 'rb') as file_name:
    engine_RUL_reader = csv.reader(file_name)
    engine_RUL = [row for row in engine_RUL_reader]
    engine_RUL = engine_RUL[flight][0]

label_to_add = [fault_type, engine_health, engine_RUL]
print label_to_add