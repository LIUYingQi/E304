# this programe is to generate a test set with random number
# we will choose 1000 randomly from fault type set and randomly choose flight from 10 to the end
# and then save 1000 test set to da txt file

import cPickle as pickle
import csv
import numpy as np
import os
import pandas as pd

# data file path
data_file_path = '/home/liuyingqi/Desktop'
time_sequence = 300

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
# process to save data
####################################################################

#  from 35 engine sample file
for file_itr in range(35):
    file = file_itr
    print file
    if file<10:
        engine = (file % 2) +1
    else:
        engine = ((file - 10) % 5) +1
    flight_min = 1
    flight_max = data_rows[file][3]
    flight_max = int(flight_max)

    path = data_file_path + '/CMAPSS_project/Graphe_data_set'
    title = 'File' + str(file+1)
    new_path = os.path.join(path,title)
    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    # clear all file in Test_data_set here to clean all pickle file
    for flight in range(flight_min,flight_max):

        graphe_set_file_name = '../Graphe_data_set/File' +str(file+1)+ '/Grapheset' + str(flight) + '.pkl'
        try:
            os.remove(graphe_set_file_name)
        except OSError:
            print 'no such files  Test set  ----   continue  '
        else:
            print 'delete old file  Test  set  ----   continue  '

        graphe_label_file_name = '../Graphe_data_set/File' +str(file+1)+ '/Graphelabel'+ str(flight) + '.pkl'
        try:
            os.remove(graphe_label_file_name)
        except OSError:
            print 'no such files  Test set  ----   continue  '
        else:
            print 'delete old file  Test  set  ----   continue  '

    # save 1 engine all flight
    for flight in range(flight_min,flight_max+1):

        # visulisation
        print '****************************************************************************************'
        print 'file : ' + str(data_rows[file][1]) + '  File : '+ str(data_rows[file]) +' flight : ' + str(flight)
        print 'flight_max : ' + str(flight_max) + '  //   flight_min : 1'
        print 'saving ----------'

        # process to save graphe dataset
        graphe_set_file_name = '../Graphe_data_set/File'+str(file+1) +'/Grapheset'+ str(flight) + '.pkl'
        with open(graphe_set_file_name,'wb') as dataset_to_save:
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
            temp = np.array([temp], dtype=np.float32)
            pickle.dump(temp, dataset_to_save)
            dataset_to_save.close()

        # process to save graphe dataset label
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
        test_set_file_name = '../Graphe_data_set/File' + str(file + 1) + '/Graphelabel'+str(flight) + '.pkl'
        with open(test_set_file_name, 'wb') as dataset_label_to_save:
            pickle.dump(np.array([label_to_add]), dataset_label_to_save)
