# this file is to do a regression to predict RUL

####################################################################################
# import and useful function
####################################################################################

import tensorflow as tf
import numpy as np
import cPickle as pickle
import time
import os

# start time
start_time = time.time()

# data file path
data_file_path = '/home/liuyingqi/Desktop'

########################################################################################
# define model
########################################################################################

# model number
model_num = 6

# pre-define value
learning_steps = 1000
input_vec_size = lstm_size = 61
time_step_size = 300
NN_input_size = 10
total_batch = 5000
layer1_size = 500
layer2_size = 250
label_size = 6
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
    output = tf.slice(output,[0,time_step_size-NN_input_size,0],[-1,NN_input_size,-1])
    output = tf.reshape(output,[-1,NN_input_size*lstm_size])
    # output : (batch_size , time_step_size * LSTM_size)

    # linear activation
    # get the last output

    layer1 = tf.matmul(output,W1) + B1

    layer2 = tf.matmul(layer1,W2) + B2

    layer_output = tf.matmul(layer2,W) + B

    # return (batch , f_x)
    return layer_output

# define X Y
X = tf.placeholder(tf.float32,[None,time_step_size,input_vec_size])
Y = tf.placeholder(tf.float32,[None,label_size])

# get lstm size and output HI

W1 = init_weight([lstm_size*NN_input_size,layer1_size])
B1 = init_weight([layer1_size])

W2 = init_weight([layer1_size,layer2_size])
B2 = init_weight([layer2_size])

W = init_weight([layer2_size,label_size])
B = init_weight([label_size])

f_x = model(X,W1,B1,W2,B2,W,B,lstm_size)

py_x = tf.nn.softmax(f_x)
cost = -tf.reduce_sum(Y * tf.log(py_x + 1e-9))
train_op = tf.train.RMSPropOptimizer().minimize(cost)
predict_op = tf.argmax( py_x , 1)

# saver
saver = tf.train.Saver()

##################################################################################
# def process
##################################################################################

def fault_str_to_num(fault_type):
    if fault_type=='Norminal':
        return [1,0,0,0,0,0]
    elif fault_type == 'Fault_Fan':
        return [0,1,0,0,0,0]
    elif fault_type == 'Fault_HPC':
        return [0,0,1,0,0,0]
    elif fault_type == 'Fault_HPT':
        return [0,0,0,1,0,0]
    elif fault_type == 'Fault_LPC':
        return [0,0,0,0,1,0]
    elif fault_type == 'Fault_LPT':
        return [0,0,0,0,0,1]

def train_part(total_batch,batch_size,time_step_size,input_vec_size,label_size,):
    print '-------------- training -----------------'

    # train for all batch for each steps but with some seperate batch
    for item_train in range(total_batch / batch_size):

        batch = np.empty((0, time_step_size, input_vec_size), dtype=np.float32)
        label = np.empty((0, 6), dtype=np.float32)

        for item_batch in range(batch_size):
            # load a batch from train set
            with open('../Train_data_set/Trainset' + str(item_train * batch_size + item_batch + 1) + '.pkl',
                      'rb') as dataset_info_to_load:
                one_batch = pickle.load(dataset_info_to_load)
            batch = np.append(batch, one_batch, axis=0)

            with open('../Train_data_set/TrainLabel' + str(item_train * batch_size + item_batch + 1)  + '.pkl',
                  'rb') as dataset_info_to_load:
                one_batch = pickle.load(dataset_info_to_load)
                one_batch = one_batch[0][0]
                one_batch = np.array([fault_str_to_num(one_batch)],dtype=np.float32)
            label = np.append(label, one_batch, axis=0)

        # visualizstion
        print 'sess run for step :' + str(step + 1) + ' /  batch : ' + str(item_train + 1)
        print 'input train batch: ' + str(batch.shape) + '----' + str(label.shape)

        # train process
        feed = {X: batch, Y: label}

        sess.run(train_op,feed_dict=feed)
        fx , cost_value = sess.run([f_x,cost], feed_dict=feed)
        print 'cost reduce sum : '+str(cost_value)
        # print fx

def test_part(label_size,time_step_size,input_vec_size,test_size,model_num):
    print '---------- test ----------------'

    # return a result to show accuracy
    batch = np.empty((0, time_step_size, input_vec_size), dtype=np.float32)
    label = np.empty((0, label_size), dtype=np.float32)

    # test with test_size sample
    for item_batch in range(test_size):
        # load a batch from test set
        with open('../Test_data_set/Testset' + str(item_batch + 1) + '.pkl', 'rb') as dataset_info_to_load:
            one_batch = pickle.load(dataset_info_to_load)
        batch = np.append(batch, one_batch, axis=0)

        # load a label from train label set
        with open('../Test_data_set/TestLabel' + str(item_batch + 1) + '.pkl','rb') as dataset_info_to_load:
            one_batch = pickle.load(dataset_info_to_load)
            one_batch = one_batch[0][0]
            one_batch = np.array([fault_str_to_num(one_batch)], dtype=np.float32)
        label = np.append(label, one_batch, axis=0)

    # calculate test accuracy and visualisation

    print 'sess test for step :' + str(step + 1)
    print 'input test batch: ' + str(batch.shape) + '----' + str(label.shape)

    feed = {X: batch, Y: label}

    f_x_label , prediction_label = sess.run([ py_x , predict_op ], feed_dict=feed)
    print f_x_label
    print prediction_label

    accuracy = 0.0
    nominal_case = 0
    prediction_nominal_case_right = 0
    prediction_nominal_case_fault = 0
    fault_case = 0
    prediction_fault_case_right = 0
    prediction_fault_case_fault = 0

    # change
    label = np.argmax(label,axis=1)

    # result register
    for i in range(test_size):
        if int(label[i]) == 0:
            # accuracy for nominal case
            nominal_case += 1
            if int(label[i]) == int(prediction_label[i]):
                accuracy += float(1) / float(test_size)
                prediction_nominal_case_right += 1
                continue
            else:
                prediction_nominal_case_fault += 1
                continue
        else:
            # accuracy for fault case
            fault_case += 1
            if int(label[i]) == int(prediction_label[i]):
                accuracy += float(1) / float(test_size)
                prediction_fault_case_right += 1
                continue
            else:
                prediction_fault_case_fault += 1
                continue

    # return accuracy
    print 'now step : ' + str(step + 1) + '  finish '
    print 'accuracy for this step : ' + str(accuracy)
    print 'accuracy for nominal case : ' + str(prediction_nominal_case_right) + ' / ' + str(nominal_case)
    print 'accuracy for fault case : ' + str(prediction_fault_case_right) + ' / ' + str(fault_case)

    # save step test info for a conclusion
    # saving
    path = data_file_path + '/CMAPSS_project/Test_result_saved'
    title = 'Model' + str(model_num)
    new_path = os.path.join(path, title)
    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    accuracy_list = []
    accuracy_nominal_case_list = []
    accuracy_fault_case_list = []

    accuracy_list.append(accuracy)
    save_result_file_name = '../Test_result_saved/Model' + str(model_num) + '/Fault_prediction_accuracy.pkl'
    try:
        os.remove(save_result_file_name)
    except OSError:
        print 'no saved file  ---  creating'
    finally:
        with open(save_result_file_name, 'wb') as result_to_save:
            pickle.dump(accuracy_list, result_to_save)
            result_to_save.close()

    accuracy_nominal_case_list.append(float(prediction_nominal_case_right) / float(nominal_case))
    save_result_file_name = '../Test_result_saved/Model' + str(
        model_num) + '/Fault_prediction_nominal_case_accuracy.pkl'
    try:
        os.remove(save_result_file_name)
    except OSError:
        print 'no saved file  ---  creating'
    finally:
        with open(save_result_file_name, 'wb') as result_to_save:
            pickle.dump(accuracy_nominal_case_list, result_to_save)
            result_to_save.close()

    accuracy_fault_case_list.append(float(prediction_fault_case_right) / float(fault_case))
    save_result_file_name = '../Test_result_saved/Model' + str(model_num) + '/Fault_prediction_fault_case_accuracy.pkl'
    try:
        os.remove(save_result_file_name)
    except OSError:
        print 'no saved file  ---  creating'
    finally:
        with open(save_result_file_name, 'wb') as result_to_save:
            pickle.dump(accuracy_fault_case_list, result_to_save)
            result_to_save.close()



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
        test_part(label_size,time_step_size,input_vec_size,test_size,model_num)

        # save model
        if step % 5 ==0:
            save_path = saver.save(sess,'../Model_saved/Model'+str(model_num)+'/Model'+str(model_num)+'_Fault_classification.tfmodel')