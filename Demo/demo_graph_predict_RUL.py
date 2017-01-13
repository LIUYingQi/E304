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

Model_num = 20

## PS: file num should test_set
for File_num in test_set:
    Step_num = 44

    # load data
    path = '../Graphe_result_saved/Model'+str(Model_num)+'/File'+str(File_num+1)+'/Step'+str(Step_num)+'_prediction_RUL.pkl'
    with open(path,'rb') as test_result:
        predict_RUL = pickle.load(test_result)

    predict_RUL = np.array(predict_RUL,dtype=np.float32)
    print ' --------------------- loading prediction RUL ------------------- '
    print ' prediction for RUL : '
    print predict_RUL

    # load real RUL
    print ' --------------------- loading real RUL ------------------- '
    if File_num+1 >10:
        print ' case : no fault '
        with open('../Data_strcture.csv') as general_info:
            reader = csv.reader(general_info)
            data_rows = [row for row in reader]
            flight_num = int(data_rows[File_num+1][3])
            print 'flight : '+str(flight_num)
            real_RUL = np.arange(flight_num,0,-1)-1
    else:
        print ' case : with fault '
        with open('../Data_strcture.csv') as general_info:
            reader = csv.reader(general_info)
            data_rows = [ row for row in reader]
            flight_num = int(data_rows[File_num+1][3])
            print ' flight : ' + str(flight_num)
            fault_flight = int(data_rows[File_num+1][4])
            real_RUL = np.append(np.ones(fault_flight-1)*(flight_num-fault_flight+1),np.arange(flight_num-fault_flight+1,0,-1)-1)

    # define : pre-plot
    step = np.arange(flight_num)+1
    step_prediction = np.arange(1,flight_num+1)

    # plot
    fig1 = plt.figure('fig1',figsize=(10,10))
    plt.plot(step,real_RUL)
    plt.plot(step_prediction,predict_RUL)
    plt.plot(step_prediction,predict_RUL - real_RUL[-len(predict_RUL):])
    plt.axhline(0,color='black')
    plt.ylim(-50,300)
    plt.xlabel('flight number')
    plt.ylabel('real RUL and predicted RUL')
    plt.title('rest useful life regression')
    plt.legend(['real RUL','predicted RUL','error'])
    plt.show()