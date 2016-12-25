# this programe is to generate a train data set with random number
# we have already randomly find 1000 as train set
# and according to the train set we generating train set

import csv
import numpy as np
import os
import cPickle as pickle
import pandas as pd

# test set :
train_set = [0,2,4,6,8,10,11,13,14,15,16,18,19,20,21,23,24,25,26,28,29,30,31,33,34]

# train set volunme
train_set_volume = 5000
time_sequence = 300

# data file path
data_file_path = '/home/liuyingqi/Desktop'

# func for change str flight_itr
def flight_itr_tostr(itr):
    if itr <= 9:
        return '00'+str(itr)
    elif itr >9 and itr <= 99:
        return '0'+str(itr)
    else:
        return str(itr)

####################################################################
# process to generate train data set
###################################################################

# clear all file in Train_data_set
try:
    os.remove('../Train_data_set/TrainsetInfo.pkl')
except OSError:
    print 'no such files  TrainsetInfo   ----   continue  '
else:
    print 'delete old file  Train set Info----   continue  '


for i in range(train_set_volume):
    train_set_file_name = '../Train_data_set/TrainLabel' + str(i + 1) + '.pkl'
    try:
        os.remove(train_set_file_name)
    except OSError:
        print 'no such files  Test set  ----   continue  '
    else:
        print 'delete old file  Test  set  ----   continue  '

for i in range(train_set_volume):
    train_set_file_name = '../Train_data_set/Trainset' + str(i + 1) + '.pkl'
    try:
        os.remove(train_set_file_name)
    except OSError:
        print 'no such files  Test set  ----   continue  '
    else:
        print 'delete old file  Test  set  ----   continue  '

# load test data set info
file_name = '../Test_data_set/TestsetInfo.csv'
with open(file_name, 'rb') as testset_info:
    testset_info_reader = csv.reader(testset_info)
    testset_info = [row for row in testset_info_reader]

# load general info from Data_strcture.csv file
with open('../Data_strcture.csv') as general_info:
    reader = csv.reader(general_info)
    data_rows = [row for row in reader]
    data_rows = data_rows[1:]

train_set_num = 0


# write all train set info in to trainsetInfo.csv
info_writer = np.empty((0,3),dtype=np.string_)
dataset_label_writer = np.empty((0, 3), dtype=np.string_)

# recurrence for generating all train set
while (train_set_num < train_set_volume):

    # randomly choose from 35 engine sample file
    file = np.random.choice(train_set)
    if file<10:
        engine = (file % 2) +1
    else:
        engine = ((file - 10) % 5) +1
    flight_min = 1
    flight_max = data_rows[file][3]
    flight_max = int(flight_max)
    flight = np.random.random_integers(flight_min, flight_max)

    # if not in test set , add this set to train set
    # visulisation
    print 'file : ' + str(data_rows[file][1]) + '  engine : '+ str(data_rows[file]) +' flight : ' + str(flight)
    print 'flight_max : ' + str(flight_max) + '  //   flight_min : 10'
    print 'saving ----------'
    info_writer = np.append(info_writer,[[str(data_rows[file][1]),str(data_rows[file][2]),str(flight)]],axis=0)

    # process to save test dataset # save in a array 32 *1000
    train_set_file_name = '../Train_data_set/Trainset' + str(train_set_num + 1) + '.pkl'
    with open(train_set_file_name, 'wb') as dataset_to_save:

        # load flight info in to array and save
        file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(
            engine) + '/Flight' + flight_itr_tostr(flight) + '_norme.csv'
        diff_file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(
            engine) + '/Flight' + flight_itr_tostr(flight) + '_diff_norme.csv'
        fuel_file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(
            engine) + '/Engine_Fuel_Effic_norme.csv'
        lifecycle_file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(
            engine) + '/Lifecycle_norme.csv'

        # load dataset
        info = pd.read_csv(file_name, engine='c')
        info = np.array(info)

        diff_info = pd.read_csv(diff_file_name, engine='c')
        diff_info = np.array(diff_info)

        info = np.append(info, diff_info, axis=1)
        array_to_add_length = info.shape[0]

        # load fuel flow and fuel efficiency
        with open(fuel_file_name) as file_to_add:
            array_to_add = np.loadtxt(file_to_add, dtype=np.float32, delimiter=',')
            fuel_effic = np.ones((array_to_add_length, 2), dtype=np.float32) * array_to_add[flight - 1, :]
            info = np.append(info, fuel_effic, axis=1)
        file_to_add.close()

        with open(lifecycle_file_name) as file_to_add:
            array_to_add = np.loadtxt(file_to_add, dtype=np.float32, delimiter=',')
            lifecycle = np.ones((array_to_add_length, 1), dtype=np.float32) * array_to_add[flight - 1]
            info = np.append(info, lifecycle, axis=1)
        file_to_add.close()

        # fill up eatch sample time setp length to 1000
        temp = np.append(np.zeros((time_sequence - len(info[:, 0]), 61), dtype=np.float32), info, axis=0)
        temp = np.array([temp],dtype=np.float32)
        pickle.dump(temp, dataset_to_save)
        dataset_to_save.close()

        # process to save train dataset label
        # write [Fault_types,health indicator,RUL]

        # read label fault_type
        file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(
            engine) + '/EngineFault_type.csv'
        with open(file_name,'rb') as file_name:
            fault_type_reader = csv.reader(file_name)
            fault_type = [row for row in fault_type_reader]
            fault_type = fault_type[flight][0]

        # read label health indicator
        file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(
            engine) + '/EngineHealth.csv'
        with open(file_name,'rb') as file_name:
            engine_health_reader = csv.reader(file_name)
            engine_health = [ row for row in engine_health_reader]
            engine_health = engine_health[flight][0]

        # read label RUL
        file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(
            engine) + '/EngineRUL.csv'
        with open(file_name,'rb') as file_name:
            engine_RUL_reader = csv.reader(file_name)
            engine_RUL = [ row for row in engine_RUL_reader]
            engine_RUL = engine_RUL[flight][0]

        label_to_add = [fault_type,engine_health,engine_RUL]
        print label_to_add

        dataset_label_writer = np.append(dataset_label_writer,[label_to_add],axis=0)

        # save label
        test_set_file_name = '../Train_data_set/TrainLabel' + str(train_set_num + 1) + '.pkl'
        with open(test_set_file_name, 'wb') as dataset_label_to_save:
            pickle.dump(np.array([label_to_add]), dataset_label_to_save)

        train_set_num += 1

with open('../Train_data_set/TrainsetLabel.pkl','wb') as dataset_label_to_save:
    pickle.dump(dataset_label_writer,dataset_label_to_save)
