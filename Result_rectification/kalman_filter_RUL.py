# this file is to use a kalman filter to rectifier RUL

import numpy
import pylab

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

## PS: file num should test_set + 1
# aaa
for File_num in test_set:
    Step_num = 44

    # load data
    path = '../Graphe_result_saved/Model'+str(Model_num)+'/File'+str(File_num+1)+'/Step'+str(Step_num)+'_prediction_RUL.pkl'
    with open(path,'rb') as test_result:
        predict_RUL = pickle.load(test_result)

    predict_RUL = np.array(predict_RUL,dtype=np.float32)
    print ' --------------------- loading prediction RUL ------------------- '
    print ' prediction for RUL : '
    # print predict_RUL

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

    # intial parameters
    n_iter = len(step)
    sz = (n_iter,)  # size of array
    x = real_RUL  # truth value
    z = predict_RUL  # observations (normal about x, sigma=0.1)

    Q = 1e-5  # process variance

    # allocate space for arrays
    xhat = numpy.zeros(sz)  # a posteri estimate of x
    P = numpy.zeros(sz)  # a posteri error estimate
    xhatminus = numpy.zeros(sz)  # a priori estimate of x
    Pminus = numpy.zeros(sz)  # a priori error estimate
    K = numpy.zeros(sz)  # gain or blending factor

    R = 1e-3  # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = predict_RUL[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = xhat[k - 1] -1 # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Pminus[k] = P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1 - K[k]) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

    pylab.figure()
    # pylab.title('predicted RUL after kalman filter')
    pylab.plot(z, 'k+', label='RUL prediction')
    pylab.plot(xhat, 'b-', label='RUL estimate')
    pylab.plot(x,'r',label='real RUL')
    pylab.axhline(0,color='g')
    pylab.ylim(-50,250)
    pylab.legend(['RUL prediction before kalman filter','RUL prediction after kalmanfilter','real RUL'],fontsize=16)
    pylab.xlabel('flight number',fontsize=16)
    pylab.ylabel('rest useful life',fontsize=16)
    pylab.show()
