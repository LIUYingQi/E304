# this file is to do a regression to predict RUL

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

# model number
model_num = 13

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
zoom_param = 10.0

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
    layer1 = tf.nn.dropout(layer1,keep_prob=0.5)

    layer2 = tf.matmul(layer1,W2) + B2
    layer2 = tf.nn.dropout(layer2,keep_prob=0.5)

    # return ( batch_size , 1 )
    return tf.matmul(layer2,W) + B , lstm.state_size

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

py_x , state_size = model(X,W1,B1,W2,B2,W,B,lstm_size)

cost = tf.reduce_sum(tf.square( py_x - Y ))
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

def train_part(total_batch,batch_size,time_step_size,input_vec_size,label_size,):
    print '-------------- training -----------------'

    # train for all batch for each steps but with some seperate batch
    for item_train in range(total_batch / batch_size):

        batch = np.empty((0, time_step_size, input_vec_size), dtype=np.float32)
        label = np.empty((0, label_size), dtype=np.float32)

        for item_batch in range(batch_size):
            # load a batch from train set
            with open('../Train_data_set/Trainset' + str(item_train * batch_size + item_batch + 1) + '.pkl',
                      'rb') as dataset_info_to_load:
                one_batch = pickle.load(dataset_info_to_load)
            batch = np.append(batch, one_batch, axis=0)

            # load a label from train label set
            with open('../Train_data_set/TrainLabel' + str(item_train * batch_size + item_batch + 1) + '.pkl',
                      'rb') as dataset_info_to_load:
                one_label = pickle.load(dataset_info_to_load)
                one_label = np.array([[one_label[0][2] ]], dtype=np.float32) * zoom_param
            label = np.append(label, one_label, axis=0)

        # visualizstion
        print 'sess run for step :' + str(step + 1) + ' /  batch : ' + str(item_train + 1)
        print 'input train batch: ' + str(batch.shape) + '----' + str(label.shape)
        # train process
        feed = {X: batch, Y: label}
        cost_value,_ = sess.run([cost,train_op], feed_dict=feed)
        print 'cost reduce mean : '+str(cost_value)

def test_part(label_size,time_step_size,input_vec_size,test_size,mean_diff,var_diff,model_num):
    print '---------- test ----------------'

    # return a result to show accuracy
    diff_prediction = np.array((0, label_size), dtype=np.int)
    batch = np.empty((0, time_step_size, input_vec_size), dtype=np.float32)
    label = np.empty((0, label_size), dtype=np.float32)

    # test with test_size sample
    for item_batch in range(test_size):
        # load a batch from test set
        with open('../Test_data_set/Testset' + str(item_batch + 1) + '.pkl', 'rb') as dataset_info_to_load:
            one_batch = pickle.load(dataset_info_to_load)
        batch = np.append(batch, one_batch, axis=0)

        # load a label from train label set
        with open('../Test_data_set/TestLabel' + str(item_batch + 1) + '.pkl', 'rb') as dataset_info_to_load:
            one_label = pickle.load(dataset_info_to_load)
            one_label = np.array([[one_label[0][2]]], dtype=np.float32) * zoom_param
        label = np.append(label, np.array(one_label, dtype=np.float32), axis=0)

    # calculate test accuracy and visualisation

    print 'sess test for step :' + str(step + 1)
    print 'input test batch: ' + str(batch.shape) + '----' + str(label.shape)

    feed = {X: batch, Y: label}

    prediction_label = np.array(sess.run(predict_op, feed_dict=feed))
    prediction_label = prediction_label / zoom_param
    label = label / zoom_param

    # result register
    for i in range(test_size):
        diff_prediction = np.append(diff_prediction, np.array(label[i] - prediction_label[i], dtype=np.int), axis=0)
        print "prediction for RUL : " + str(np.round(prediction_label[i])) + " //  real RUL : " + str(label[i] )

    # return accuracy
    mean_diff_to_append = np.mean(diff_prediction)
    var_diff_to_append = np.var(diff_prediction)

    print 'mean for difference : ' + str(mean_diff_to_append)
    print 'derivation for difference : ' + str(var_diff_to_append)

    # save step test info for a conclusion
    # saving
    path = data_file_path + '/CMAPSS_project/Test_result_saved'
    title = 'Model' + str(model_num)
    new_path = os.path.join(path, title)
    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    mean_diff.append(mean_diff_to_append)
    save_result_file_name = '../Test_result_saved/Model'+str(model_num)+'/mean_diff_RUL.pkl'
    try:
        os.remove(save_result_file_name)
    except OSError:
        print 'no saved file  ---  creating'
    finally:
        with open(save_result_file_name, 'wb') as result_to_save:
            pickle.dump(mean_diff, result_to_save)
            result_to_save.close()

    var_diff.append(var_diff_to_append)
    save_result_file_name = '../Test_result_saved/Model'+str(model_num)+'/var_diff_RUL.pkl'
    try:
        os.remove(save_result_file_name)
    except OSError:
        print 'no saved file  ---  creating'
    finally:
        with open(save_result_file_name, 'wb') as result_to_save:
            pickle.dump(var_diff, result_to_save)
            result_to_save.close()


def draw_graphe_part(time_step_size,input_vec_size,label_size,model_num):

    if (step+1)%1==0 and step>0:
        print '------------ save model predict result each 5 steps ----------------'

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
                    one_label = np.array([[one_label[0][2]]], dtype=np.float32)* zoom_param
                label = np.append(label, np.array(one_label, dtype=np.float32), axis=0)

            # predict to show this stpe result
            feed = {X: batch, Y: label}

            prediction_label = sess.run(predict_op, feed_dict=feed)

            # transformation
            label = label.flatten()
            prediction_label = np.array(prediction_label,dtype=np.float32)
            prediction_label = prediction_label.flatten()
            prediction_label = prediction_label / zoom_param

            # saving
            path = data_file_path + '/CMAPSS_project/Graphe_result_saved'
            title = 'Model'+str(model_num)+'/File' + str(file + 1)
            new_path = os.path.join(path, title)
            if not os.path.isdir(new_path):
                os.makedirs(new_path)

            save_result_file_name = '../Graphe_result_saved/Model'+str(model_num)+'/File'+ str(file + 1) +'/Step'+str(step+1)+'_prediction_RUL.pkl'
            with open(save_result_file_name,'wb') as result_to_save:
                pickle.dump(prediction_label, result_to_save)
                result_to_save.close()
    else:
        return

###################################################################################
# lunch session
###################################################################################

with tf.Session() as sess:

    # model value initialization
    tf.initialize_all_variables().run()

    # learning_steps
    for step in range(learning_steps):

        print '                                                            '
        print '############################################################'
        print '                                                            '

        # show time
        end_time = time.time()
        cost_time = end_time - start_time
        print ' total run time : ' + str(cost_time)

        # show step info
        print 'step : ' + str(step+1) + ' in ' + str(learning_steps) + ' learning steps'

        # train part
        train_part(total_batch,batch_size,time_step_size,input_vec_size,label_size)

        # test part
        test_part(label_size,time_step_size,input_vec_size,test_size,mean_diff,var_diff,model_num)

        # draw graphe part
        draw_graphe_part(time_step_size,input_vec_size,label_size,model_num)

        # save model
        save_path = saver.save(sess,'../Model_saved/Model'+str(model_num)+'/Model'+str(model_num)+'_RUL.tfmodel')