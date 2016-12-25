import cPickle as pickle
import pandas as pd
import csv
import numpy as np
import os

# test set :

# test_set = [1,3,5,9,12,17,22,27,32,34]
# test_set = [0,2,4,8,10,11,13,15,16,18,19,20,21,23,24,25,28,30,31,33]

# test_set = [0,2,5,9,12,17,22,27,32,34]
# train_set = [1,3,4,8,10,11,13,15,16,18,19,20,21,23,24,25,28,30,31,33]

test_set = [0,2,5,8,13,17,22,25,28,34]
train_set = [1,3,4,9,10,11,12,15,16,18,19,20,21,23,24,27,30,31,32,33]

probability_set = [0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1,0.1]

# pre-define value
test_set_volume = 1000
time_sequence = 300

# data file path
data_file_path = '/home/liuyingqi/Desktop'

# load general info from Data_strcture.csv file
with open('../Data_strcture.csv') as general_info:
    reader = csv.reader(general_info)
    data_rows = [ row for row in reader]
    data_rows = data_rows[1:]

# func for change str flight_itr
def flight_itr_tostr(itr):
    if itr <= 9:
        return '00'+str(itr)
    elif itr >9 and itr <= 99:
        return '0'+str(itr)
    else:
        return str(itr)

#####################################################################
# process to save test dataset info
####################################################################

# clear all file in Test_data_set here to clean all pickle file
try:
    os.remove('../Test_data_set_without_2edge/TestsetInfo.pkl')
except OSError:
    print 'no such files  TestsetInfo   ----   continue  '
else:
    print 'delete old file  Test set Info----   continue  '

for i in range(test_set_volume):
    test_set_file_name = '../Test_data_set_without_2edge/TestLabel' + str(i+1) + '.pkl'
    try:
        os.remove(test_set_file_name)
    except OSError:
        print 'no such files  Test set  ----   continue  '
    else:
        print 'delete old file  Test  set  ----   continue  '

for i in range(test_set_volume):
    test_set_file_name = '../Test_data_set_without_2edge/Testset' + str(i+1) + '.pkl'
    try:
        os.remove(test_set_file_name)
    except OSError:
        print 'no such files  Test set  ----   continue  '
    else:
        print 'delete old file  Test  set  ----   continue  '

# write all test set info and label in to list
info_writer = np.empty((0,3),dtype=np.string_)

# recurrence for generating all test set
for test_num in range(test_set_volume):

    # randomly choose from 35 engine sample file
    file = np.random.choice(test_set,p=probability_set)
    print file
    if file<10:
        engine = (file % 2) +1
    else:
        engine = ((file - 10) % 5) +1
    flight_min = 1
    flight_max = data_rows[file][3]
    flight_max = int(flight_max)
    flight = np.random.random_integers(flight_min, flight_max)

    # visulisation
    print 'file : ' + str(data_rows[file][1]) + '  engine : '+ str(data_rows[file]) +' flight : ' + str(flight)
    print 'flight_max : ' + str(flight_max) + '  //   flight_min : 1'
    print 'saving ----------'

    # save test data info
    info_writer = np.append(info_writer,np.array([[str(data_rows[file][1]),data_rows[file][2],flight]]),axis=0)

    # process to save test dataset # save in a array 32 *1000
    test_set_file_name = '../Test_data_set_without_2edge/Testset' + str(test_num+1) + '.pkl'
    with open(test_set_file_name,'wb') as dataset_to_save:

        # load flight info in to array and save
        file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(engine) + '/Flight' + flight_itr_tostr(flight) + '_norme.csv'
        diff_file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(engine) + '/Flight' + flight_itr_tostr(flight) + '_diff_norme.csv'
        fuel_file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(engine) + '/Engine_Fuel_Effic_norme.csv'
        lifecycle_file_name = data_file_path + '/CMAPSS_dataset/' + str(data_rows[file][1]) + '/Engine0' + str(engine) + '/Lifecycle_norme.csv'

        # load dataset
        info = pd.read_csv(file_name,engine='c')
        info = np.array(info)

        diff_info = pd.read_csv(diff_file_name,engine='c')
        diff_info = np.array(diff_info)

        info = np.append(info,diff_info,axis=1)
        array_to_add_length = info.shape[0]

        # load fuel flow and fuel efficiency
        with open(fuel_file_name) as file_to_add:
            array_to_add = np.loadtxt(file_to_add, dtype=np.float32, delimiter=',')
            fuel_effic = np.ones((array_to_add_length,2),dtype=np.float32) * array_to_add[flight-1,:]
            info = np.append(info,fuel_effic,axis=1)
        file_to_add.close()

        with open(lifecycle_file_name) as file_to_add:
            array_to_add = np.loadtxt(file_to_add, dtype=np.float32, delimiter=',')
            lifecycle = np.ones((array_to_add_length,1),dtype=np.float32) * array_to_add[flight-1]
            info = np.append(info,lifecycle,axis=1)
        file_to_add.close()

        print file_name + '   ---   OK'

        # fill up eatch sample time setp length to 1000
        temp = np.append(np.zeros((time_sequence-len(info[:,0]),61),dtype=np.float32),info,axis=0)
        temp = np.array([temp],dtype=np.float32)
        pickle.dump(temp,dataset_to_save)
        dataset_to_save.close()

    # process to save test dataset label
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

    # save label
    test_set_file_name = '../Test_data_set_without_2edge/TestLabel' + str(test_num + 1) + '.pkl'
    with open(test_set_file_name, 'wb') as dataset_label_to_save:
        pickle.dump(np.array([label_to_add]), dataset_label_to_save)

with open('../Test_data_set_without_2edge/TestsetInfo.pkl','wb') as dataset_info_to_save:
    pickle.dump(info_writer,dataset_info_to_save)