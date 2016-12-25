# this programme is to demonstrate predict result for each case

import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import csv

# test_set = [1,3,5,7,9,12,17,22,27,32]
# train_set = [0,2,4,6,8,10,11,13,14,15,16,18,19,20,21,23,24,25,26,28,29,30,31,33,34]

# test_set = [1,3,5,9,12,17,22,27,32,34]
# test_set = [0,2,4,8,10,11,13,15,16,18,19,20,21,23,24,25,28,30,31,33]

# test_set = [0,2,5,9,12,17,22,27,32,34]
# train_set = [1,3,4,8,10,11,13,15,16,18,19,20,21,23,24,25,28,30,31,33]

test_set = [0,2,5,8,13,17,22,25,28,34]
train_set = [1,3,4,9,10,11,12,15,16,18,19,20,21,23,24,27,30,31,32,33]

Model_num = 6

## PS: file num should test_set
for File_num in test_set:
    Step_num = 170
    # load data

    path = '../Graphe_result_saved/Model'+str(Model_num)+'/File'+str(File_num+1)+'/Step'+str(Step_num)+'_prediction_HI.pkl'
    with open(path,'rb') as test_result:
        predict_HI = pickle.load(test_result)
    predict_HI = np.array(predict_HI,dtype=np.float32)

    print ' --------------------- loading prediction HI ------------------- '
    print ' prediction for HI : '
    print predict_HI
    print predict_HI.size

    # load real RUL

    print ' --------------------- loading real HI ------------------- '
    path = '/home/liuyingqi/Desktop/CMAPSS_dataset'
    # load general info from Data_strcture.csv file
    with open('../Data_strcture.csv') as general_info:
        reader = csv.reader(general_info)
        data_rows = [ row for row in reader]

    flight_num = data_rows[File_num+1][3]

    if File_num+1 >10:
        print ' case : no fault '
        with open(path+'/'+data_rows[File_num+1][1]+'/Engine0'+data_rows[File_num+1][2]+'/EngineHealth.csv') as general_info:
            reader = csv.reader(general_info)
            data_rows = [row for row in reader]
            data_rows = data_rows[1:]
            data_rows = np.array(data_rows)
            data_rows = data_rows.flatten()

            print data_rows.size
    else:
        print ' case : with fault '
        with open(path+'/'+data_rows[File_num+1][1]+'/Engine0'+data_rows[File_num+1][2]+'/EngineHealth.csv') as general_info:
            reader = csv.reader(general_info)
            data_rows = [row for row in reader]
            data_rows = data_rows[1:]
            data_rows = np.array(data_rows)
            data_rows = data_rows.flatten()
            print data_rows.size

    # define : pre-plot

    step = np.arange(int(flight_num))+1
    predict_step = np.arange(int(flight_num))+1
    print step
    print predict_step

    # plot
    fig1 = plt.figure('fig1')
    plt.title(str(Step_num)+'   '+str(File_num))
    plt.plot(step,data_rows)
    plt.plot(predict_step,predict_HI)

    plt.xlabel('step ')
    plt.ylabel('difference prediction and real HI')
    plt.show()