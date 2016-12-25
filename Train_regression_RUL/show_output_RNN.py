# this file is to show what's output of RNN

####################################################################################
# import and useful function
####################################################################################

import tensorflow as tf
import numpy as np
import cPickle as pickle
import time
import csv
import os

# start time
start_time = time.time()

# data file path
data_file_path = '/home/liuyingqi/Desktop'

########################################################################################
# define model
########################################################################################

test_set = [0,2,5,8,13,17,22,25,28,34]
train_set = [1,3,4,9,10,11,12,15,16,18,19,20,21,23,24,27,30,31,32,33]

# model number
model_num = 16

# pre-define value
learning_steps = 1000
input_vec_size = lstm_size = 61
time_step_size = 300
NN_input_size = 10
total_batch = 5000
layer1_size = 500
layer2_size = 250
label_size = 1
batch_size = 100  # change with each time generating train set
test_size = 1000  # not need to be 1000
zoom_param = 1000.0

# test result list
mean_diff = []
var_diff = []

# initial for weight
def init_weight(shape):
    return tf.Variable(tf.random_normal(shape,stddev=1.0))

# define model
def model(X,W1,B1,W2,B2,W,B,lstm_size):
    # X,input shape: (batch_size,time_step_size,input_vec_size)
    XT = tf.transpose(X,[1,0,2])
    # XT shape : (time_step_size,batch_size,input_vec_size)
    XR = tf.reshape(XT,[-1,lstm_size])
    # XR shape : (time_step_size * batch_size ,input_vec_size)
    X_split = tf.split(0,time_step_size,XR)
    # sequence_num array with each array(batch_size,input_vec_size )

    # defin lstm cell
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size,forget_bias=1.0,state_is_tuple=True)
    # get lstm cell output
    output , _states = tf.nn.rnn(lstm,X_split,dtype=tf.float32)
    # output : time_step_size arrays with each array (batch_size , LSTM_size) so (time_step_size , batch_size , LSTM_size )

    output = tf.transpose(output,perm=[1,0,2])
    output = tf.slice(output,[0,time_step_size-NN_input_size-6,0],[-1,NN_input_size,-1])
    output = tf.reshape(output,[-1,NN_input_size*lstm_size])
    # output : (batch_size , time_step_size * LSTM_size)

    # linear activation
    # get the last output

    layer1 = tf.matmul(output,W1) + B1

    layer2 = tf.matmul(layer1,W2) + B2

    # return ( batch_size , 1 )
    return tf.matmul(layer2,W) + B , lstm.state_size , output

# define X Y
X = tf.placeholder(tf.float32,[None,time_step_size,input_vec_size])
Y = tf.placeholder(tf.float32,[None,1])

# get lstm size and output RUL

W1 = init_weight([lstm_size*NN_input_size,layer1_size])
B1 = init_weight([layer1_size])

W2 = init_weight([layer1_size,layer2_size])
B2 = init_weight([layer2_size])

W = init_weight([layer2_size,label_size])
B = init_weight([label_size])

py_x , state_size , output = model(X,W1,B1,W2,B2,W,B,lstm_size)

cost = tf.reduce_mean(tf.square( py_x - Y ))
train_op = tf.train.AdamOptimizer().minimize(cost)
predict_op = tf.round( py_x )

# saver
saver = tf.train.Saver()

####################################################################################
# process function
###################################################################################

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

##################################################################################
# def process
##################################################################################


def save_output_part(time_step_size,input_vec_size,label_size,model_num):

    print '------------ print RNN output ----------------'

    # for each 35 case save a set of data
    for file_itr in range(35):
        file = file_itr
        print 'now file :' + str(file+1)
        if file < 10:
            engine = (file % 2) + 1
        else:
            engine = ((file - 10) % 5) + 1
        flight_min = 1
        flight_max = data_rows[file][3]
        flight_max = int(flight_max)

        batch = np.empty((0, time_step_size, input_vec_size), dtype=np.float32)
        label = np.empty((0, label_size), dtype=np.float32)

        # load file in graphe_data_set here to save result
        for flight in range(flight_min, flight_max+1):

            # load a batch from test set
            with open('../Graphe_data_set/File' + str(file_itr + 1) + '/Grapheset'+str(flight)+'.pkl', 'rb') as dataset_info_to_load:
                one_batch = pickle.load(dataset_info_to_load)
            batch = np.append(batch, one_batch, axis=0)

            # load a label from train label set
            with open('../Graphe_data_set/File' + str(file_itr + 1) + '/Graphelabel' + str(flight) + '.pkl', 'rb') as dataset_info_to_load:
                one_label = pickle.load(dataset_info_to_load)
                one_label = np.array([[one_label[0][1]]], dtype=np.float32)* zoom_param
            label = np.append(label, np.array(one_label, dtype=np.float32), axis=0)

        # predict to show this stpe result
        feed = {X: batch, Y: label}

        RNN_output = sess.run(output , feed_dict=feed)
        with open('../Graphe_result_saved/Model'+str(model_num)+'/File' + str(file_itr + 1) + '/RNN_output_RUL.pkl','wb') as RNN_output_to_save:
            pickle.dump(RNN_output,RNN_output_to_save)


###################################################################################
# lunch session
###################################################################################

with tf.Session() as sess:

    # model value restore
    # load model
    saver.restore(sess,'../Model_saved/Model'+str(model_num)+'/Model'+str(model_num)+'_RUL.tfmodel')
    print '------------- model restored ------------'

    # print RNN output
    save_output_part(time_step_size, input_vec_size, label_size, model_num)